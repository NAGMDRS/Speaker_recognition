import webrtcvad
import torchaudio
import numpy as np
import soundfile as sf
import os
from pydub import AudioSegment

vad = webrtcvad.Vad(3)

def preprocess_audio(audio_path, target_sr=16000):
    """Ensure the audio is 16kHz, 16-bit PCM, mono"""
    print(f"[INFO] Preprocessing {audio_path}...")

    audio = AudioSegment.from_file(audio_path)
    # print(f" - Original Sample Rate: {audio.frame_rate} Hz")
    # print(f" - Channels: {audio.channels}")
    # print(f" - Sample Width: {audio.sample_width * 8} bits")

    audio = audio.set_frame_rate(target_sr).set_sample_width(2).set_channels(1)
    processed_path = audio_path.replace(".wav", "_processed.wav")
    audio.export(processed_path, format="wav")

    print(f"[INFO] Converted to 16kHz, 16-bit PCM, mono -> {processed_path}")
    return processed_path

def get_speech_segments(audio_path, output_dir,file_name,sr=16000, frame_length=30):
    """Detect speech segments and save them as separate WAV files"""

    audio_path = preprocess_audio(audio_path, sr)

    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.numpy().squeeze()

    # print(f"[DEBUG] Processing file: {audio_path}")
    # print(f" - Loaded Sample Rate: {sample_rate} Hz")
    # print(f" - Audio Length: {len(waveform) / sample_rate:.2f} sec")

    if sample_rate != sr:
        print("[ERROR] Sample rate mismatch! Resample to 16kHz before processing.")
        return

    waveform = (waveform * 32768).astype(np.int16)

    valid_frame_sizes = {160, 320, 480}
    frame_size = int(sr * frame_length / 1000)

    if frame_size not in valid_frame_sizes:
        print(f"[ERROR] Invalid frame size: {frame_size} samples")
        return

    print(f"[INFO] Frame Size: {frame_size} samples ({frame_length} ms)")

    segments = []
    current_segment = []

    for i in range(0, len(waveform) - frame_size, frame_size):
        frame = waveform[i: i + frame_size]

        frame_bytes = frame.tobytes()

        if len(frame_bytes) != frame_size * 2:
            print(f"[ERROR] Frame buffer size mismatch: {len(frame_bytes)} bytes (expected {frame_size * 2})")
            continue

        is_speech = vad.is_speech(frame_bytes, sr)

        if is_speech:
            print(f" - Speech detected at {i / sr:.2f} sec")
            current_segment.append(frame)
        elif current_segment:
            segments.append(np.concatenate(current_segment))
            current_segment = []

    if current_segment:
        segments.append(np.concatenate(current_segment))

    min_word_length = 0.2
    max_word_length = 5.0
    filtered_segments = [seg for seg in segments if min_word_length * sr < len(seg) < max_word_length * sr]

    print(f"[DEBUG] Filtered Segments: {filtered_segments}")
    os.makedirs(output_dir, exist_ok=True)
    for i, seg in enumerate(filtered_segments):
        heppi=file.replace(".wav", "")
        output_file = f"{output_dir}/{[heppi]}word_{i}.wav"
        print(output_file)
        sf.write(output_file, seg, samplerate=sr)
        print(f"[SAVED] {output_file}")

    print("[INFO] Processing complete.\n")


def process_and_segment():
    directory = "F:/Datasets/IndianVoxCeleb/vox_indian"
    folders = os.listdir(directory)

    for speakerid in folders:
        for folder in os.listdir(directory + "/" + speakerid):
            for file in os.listdir(directory + "/" + speakerid + "/" + folder):
                if not (file.endswith("_processed.wav")):
                    file_path = directory + "/" + speakerid + "/" + folder + "/" + file
                    output_dir = "F:/Datasets/IndianVoxCeleb/vox_indian_split/"+speakerid+"/"+folder
                    get_speech_segments(file_path, sr=16000, output_dir=output_dir, file_name=file)