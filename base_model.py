import math, torch
import torch.nn as nn
from cfg import cfg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def mel_filter_bank(n_mels=80, n_fft=512, sample_rate=16000, f_min=20, f_max=7600):
    """
    Creates a Mel filter bank matrix to convert a power spectrogram into a Mel spectrogram.

    This function calculates triangular filters spaced according to the Mel scale and 
    maps linear frequency bins (from the FFT) into Mel-scaled frequency bins.

    Args:
        n_mels (int): Number of Mel filters.
        n_fft (int): Number of FFT components.
        sample_rate (int): Audio sample rate.
        f_min (float): Minimum frequency to include in the Mel scale.
        f_max (float): Maximum frequency to include in the Mel scale.

    Returns:
        torch.Tensor: Mel filter bank matrix of shape (n_mels, n_fft // 2 + 1).
    """
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
    """
    Computes the Mel spectrogram from a raw waveform using a custom implementation.

    Uses short-time Fourier transform (STFT) followed by a Mel filter bank projection.
    """
    def __init__(self, sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_mels=80):
        """
        Initializes the parameters for Mel spectrogram computation.

        Args:
            sample_rate (int): Audio sample rate in Hz.
            n_fft (int): Number of FFT bins.
            win_length (int): Length of each STFT window.
            hop_length (int): Hop size between windows.
            n_mels (int): Number of Mel filters.
        """
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.window = torch.hamming_window(win_length).to(device)
        self.mel_filter = mel_filter_bank(n_mels, n_fft, sample_rate)

    def forward(self, waveform):
        """
        Computes the Mel spectrogram from a given waveform.

        Args:
            waveform (Tensor): Audio waveform of shape (batch, time).

        Returns:
            Tensor: Mel spectrogram of shape (batch, n_mels, frames).
        """
        stft = torch.stft(
            waveform.to(device), n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=self.window,
            return_complex=True
        )
        power_spec = torch.abs(stft) ** 2
        mel_spec = torch.matmul(self.mel_filter, power_spec)
        return mel_spec


class SEModule(nn.Module):
    """
    Implements the Squeeze-and-Excitation (SE) attention mechanism.

    Dynamically recalibrates channel-wise feature responses by explicitly modelling
    interdependencies between channels.
    """
    def __init__(self, channels, bottleneck=128):
        """
        Args:
            channels (int): Number of input and output channels.
            bottleneck (int): Number of channels in the bottleneck layer.
        """
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, input):
        """
        Applies channel-wise attention.

        Args:
            input (Tensor): Input feature map of shape (batch, channels, time).

        Returns:
            Tensor: Recalibrated feature map of same shape.
        """
        x = self.se(input)
        return input * x


class Bottle2neck(nn.Module):
    """
    A Res2Net-based bottleneck block for multi-scale feature extraction.

    Splits channels into groups and applies convolution to each sequentially to capture
    diverse receptive fields. Includes SE attention for channel weighting.
    """
    def __init__(self, inplanes, planes, kernel_size=3, dilation=2, scale=8):
        """
        Args:
            inplanes (int): Number of input channels.
            planes (int): Number of output channels.
            kernel_size (int): Kernel size for convolutions.
            dilation (int): Dilation rate for temporal convolution.
            scale (int): Number of feature groups to split input into.
        """
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

        self.convs = nn.ModuleList(convs).to(device)
        self.bns = nn.ModuleList(bns).to(device)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1).to(device)
        self.bn3 = nn.BatchNorm1d(planes).to(device)
        self.relu = nn.ReLU().to(device)
        self.width = width
        self.se = SEModule(planes).to(device)

    def forward(self, x):
        """
        Applies grouped convolutions followed by residual and SE attention.

        Args:
            x (Tensor): Input tensor of shape (batch, channels, time).

        Returns:
            Tensor: Output tensor of same shape after bottleneck processing.
        """
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
    """
    ECAPA-TDNN model for extracting speaker embeddings from raw audio.

    Integrates multi-layer bottleneck blocks, attentive statistical pooling,
    and dimensionality reduction layers to output a fixed-size embedding.
    """
    def __init__(self, C):
        """
        Args:
            C (int): Number of channels for intermediate convolution layers.
        """
        super(ECAPA_TDNN, self).__init__()
        self.torchfbank = CustomMelSpectrogram().to(device)
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

    def forward(self, x, aug=False):
        """
        Computes a speaker embedding from an audio waveform.

        Args:
            x (Tensor): Input waveform of shape (batch, time).
            aug (bool): Whether to apply SpecAugment.

        Returns:
            Tensor: Speaker embedding of shape (batch, 256).
        """
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
        global_x = torch.cat((x, torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                              torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)), dim=1)
        w = self.attention(global_x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x ** 2) * w, dim=2) - mu ** 2).clamp(min=1e-4))
        x = torch.cat((mu, sg), 1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)
        return x

    def apply_spec_augment(self, x, freq_mask=10, time_mask=10):
        """
        Applies SpecAugment for data augmentation by masking frequency and time bands.

        Args:
            x (Tensor): Mel spectrogram input of shape (batch, channels, time).
            freq_mask (int): Maximum width of frequency masking.
            time_mask (int): Maximum width of time masking.

        Returns:
            Tensor: Augmented Mel spectrogram.
        """
        batch_size, channels, time_steps = x.shape

        for _ in range(2):
            f = torch.randint(0, freq_mask, (1,)).item()
            f0 = torch.randint(0, channels - f, (1,)).item()
            x[:, f0:f0 + f, :] = 0

        for _ in range(2):
            t = torch.randint(0, time_mask, (1,)).item()
            t0 = torch.randint(0, time_steps - t, (1,)).item()
            x[:, :, t0:t0 + t] = 0

        return x
