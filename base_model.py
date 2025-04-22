import math, torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def mel_filter_bank(n_mels=80, n_fft=512, sample_rate=16000, f_min=20, f_max=7600):
    mel_scale = torch.linspace(
        1125 * math.log(1 + f_min / 700.0),
        1125 * math.log(1 + f_max / 700.0),
        n_mels + 2,
    ).to(device)
    hz_scale = 700 * (torch.exp(mel_scale / 1125) - 1)
    bin_edges = torch.floor((n_fft + 1) * hz_scale / sample_rate).long()
    filter_bank = torch.zeros(n_mels, n_fft // 2 + 1, device=device)

    for i in range(1, n_mels + 1):
        left, center, right = bin_edges[i - 1], bin_edges[i], bin_edges[i + 1]
        for j in range(left, center):
            filter_bank[i - 1, j] = (j - left) / (center - left)
        for j in range(center, right):
            filter_bank[i - 1, j] = (right - j) / (right - center)
    return filter_bank


class CustomMelSpectrogram(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=80):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.window = torch.hamming_window(win_length).to(device)
        self.mel_filter = mel_filter_bank(n_mels, n_fft, sample_rate)

    def forward(self, waveform):
        stft = torch.stft(
            waveform.to(device), n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=self.window,
            return_complex=True
        )
        power_spec = torch.abs(stft) ** 2
        mel_spec = torch.matmul(self.mel_filter, power_spec)
        return mel_spec


class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, input):
        x = self.se(input)
        return input * x


class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, dilation=2, scale=8):
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1).to(device)
        self.bn1 = nn.BatchNorm1d(width * scale).to(device)
        self.nums = scale - 1
        convs, bns = [], []
        num_pad = math.floor(kernel_size / 2) * dilation
        for i in range(self.nums):
            convs.append(
                nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad).to(device))
            bns.append(nn.BatchNorm1d(width).to(device))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1).to(device)
        self.bn3 = nn.BatchNorm1d(planes).to(device)
        self.relu = nn.ReLU().to(device)
        self.se = SEModule(planes).to(device)
        self.width = width

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        out = self.se(out)
        out += residual
        return out


class ECAPA_TDNN(nn.Module):
    def __init__(self, C):
        super(ECAPA_TDNN, self).__init__()
        self.torchfbank = CustomMelSpectrogram(sample_rate=16000, n_fft=512,
                                               win_length=400, hop_length=160, n_mels=80).to(device)
        self.conv1 = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2).to(device)
        self.relu = nn.ReLU().to(device)
        self.bn1 = nn.BatchNorm1d(C).to(device)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8).to(device)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8).to(device)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8).to(device)
        self.layer4 = nn.Conv1d(3 * C, 1536, kernel_size=1).to(device)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
        ).to(device)
        self.bn5 = nn.BatchNorm1d(3072).to(device)
        self.fc6 = nn.Linear(3072, 256).to(device)
        self.bn6 = nn.BatchNorm1d(256).to(device)

    def forward(self, x, aug=False, return_attention=False):
        x = x.to(device)
        with torch.no_grad():
            x = self.torchfbank(x) + 1e-6
            x = x.log()
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug:
                x = self.apply_spec_augment(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)
        x = self.layer4(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x)
        t = x.size()[-1]
        global_x = torch.cat((
            x,
            torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
            torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)
        ), dim=1)
        w = self.attention(global_x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))
        x = torch.cat((mu, sg), 1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)
        if return_attention:
            return x, w
        return x

    def apply_spec_augment(self, x, freq_mask=10, time_mask=10):
        batch_size, channels, time_steps = x.shape
        # Frequency Masking
        for _ in range(2):
            f = torch.randint(0, freq_mask, (1,)).item()
            f0 = torch.randint(0, channels - f, (1,)).item()
            x[:, f0:f0 + f, :] = 0
        # Time Masking
        for _ in range(2):
            t = torch.randint(0, time_mask, (1,)).item()
            t0 = torch.randint(0, time_steps - t, (1,)).item()
            x[:, :, t0:t0 + t] = 0
        return x


def visualize_attention_amplitude(model, waveform):
    model.eval()
    with torch.no_grad():
        emb, w = model(waveform.unsqueeze(0), return_attention=True)
    mel_spec = model.torchfbank(waveform.unsqueeze(0))
    amplitude = mel_spec.sum(dim=1).squeeze(0).cpu().numpy()
    attention = w.mean(dim=1).squeeze(0).cpu().numpy()
    combined = np.vstack([amplitude, attention])
    plt.figure()
    plt.imshow(combined, aspect='auto', origin='lower')
    plt.yticks([0, 1], ['Amplitude', 'Attention'])
    plt.xlabel('Time Frames')
    plt.title('Attention vs Amplitude Heatmap')
    plt.show()


# Example usage:
# uncomment this if want to use: model = ECAPA_TDNN(C=512).to(device)
# waveform = torch.randn(16000)
# visualize_attention_amplitude(model, waveform)
