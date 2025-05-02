import torch
import matplotlib.pyplot as plt
from main_model import ECAPAModel
from helperFiles.dataLoader import train_loader
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visualize_attention_from_loader(model, dataloader, save_path="attention_plot.png"):
    model.eval()

    for batch in dataloader:
        waveform, label = batch
        waveform = waveform.to(device)

        with torch.no_grad():
            min_length = 512
            if waveform.shape[-1] < min_length:
                pad_amount = min_length - waveform.shape[-1]
                padding_tuple = (0, pad_amount, 0, 0)
                waveform = F.pad(waveform, padding_tuple, mode="reflect")

            mel = model.speaker_encoder.torchfbank(waveform) + 1e-6
            mel = mel.log()
            mel = mel - torch.mean(mel, dim=-1, keepdim=True)

            x = model.speaker_encoder.conv1(mel)
            x = model.speaker_encoder.relu(x)
            x = model.speaker_encoder.bn1(x)

            x1 = model.speaker_encoder.layer1(x)
            x2 = model.speaker_encoder.layer2(x + x1)
            x3 = model.speaker_encoder.layer3(x + x1 + x2)
            x = model.speaker_encoder.layer4(torch.cat((x1, x2, x3), dim=1))
            x = model.speaker_encoder.relu(x)

            t = x.size()[-1]  # Number of time frames in the feature map
            global_x = torch.cat((
                x,
                torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t),
                torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, t)
            ), dim=1)

            w = model.speaker_encoder.attention(global_x)
            avg_attn = w.mean(dim=1)

            avg_attn_np = avg_attn.squeeze().cpu().numpy()
            mel_np = mel.squeeze().cpu().numpy()
            waveform_np = waveform.squeeze().cpu().numpy()

            fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

            axs[0].plot(waveform_np)
            axs[0].set_title("Input Waveform")
            axs[0].set_ylabel("Amplitude")

            img = axs[1].imshow(mel_np, aspect='auto', origin='lower', cmap='viridis',
                                extent=[0, waveform_np.shape[0], 0, mel_np.shape[0]])

            time_frames_indices = np.arange(avg_attn_np.shape[0])
            hop_length = model.speaker_encoder.torchfbank.hop_length  # Get hop length (160)
            audio_sample_indices = time_frames_indices * hop_length

            min_attn = np.min(avg_attn_np)
            max_attn = np.max(avg_attn_np)

            if max_attn == min_attn:
                scaled_attention = np.full_like(avg_attn_np, mel_np.shape[0] * 0.1)
            else:
                scaled_attention = (avg_attn_np - min_attn) / (max_attn - min_attn) * mel_np.shape[0]

            axs[1].plot(audio_sample_indices, scaled_attention, color='red', label='Attention', linewidth=2)

            axs[1].legend()
            axs[1].set_title("Mel Spectrogram + Attention")
            axs[1].set_xlabel("Time Frame (Audio Samples)")  # Added clarification
            axs[1].set_ylabel("Mel Channel Index / Attention Scale")  # Added clarification
            plt.colorbar(img, ax=axs[1], format='%+2.0f dB')

            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

        break


if __name__ == "__main__":

    model = ECAPAModel(C=1024, m=0.2, s=30, n_class=24).to(device)
    try:
        model.load_state_dict(
            torch.load("/home/btech10154.22/Speaker_recognition-main/exps/exp1/model4_0005.model", map_location=device))
    except RuntimeError as e:
        print(f"Error loading model state_dict: {e}")
        print("Attempting to load with strict=False (ignoring mismatched keys in feature extractor)")
        model.load_state_dict(
            torch.load("/home/btech10154.22/Speaker_recognition-main/exps/exp1/model4_0005.model", map_location=device),
            strict=False)

    dataset = train_loader()
    trainDataLoader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        persistent_workers=False
    )

    visualize_attention_from_loader(model, trainDataLoader,
                                    save_path="/home/btech10154.22/Speaker_recognition-main/Images/ecapa_attention_output.png")
