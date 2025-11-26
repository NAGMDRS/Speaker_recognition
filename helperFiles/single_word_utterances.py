import webrtcvad
import torchaudio
import numpy as np
import soundfile as sf
import os
from pydub import AudioSegment
import multiprocessing
from tqdm import tqdm

def preprocess_audio_in_memory(audio_path, target_sr=16000):
    """
    CHANGED: Convert audio and return it as a numpy array instead of saving a file.

    Args:
        audio_path (str): Path to the input audio file.
        target_sr (int): Target sample rate in Hz.

    Returns:
        tuple: (np.array of int16 samples, sample_rate)
    """
    try:
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_frame_rate(target_sr).set_sample_width(2).set_channels(1)

        # Get samples as a numpy array
        samples = np.array(audio.get_array_of_samples()).astype(np.int16)

        return samples, audio.frame_rate
    except Exception as e:
        print(f"[ERROR] Failed to preprocess {audio_path}: {e}")
        return None, 0


def get_speech_segments(vad, waveform_int16, sample_rate, output_dir, file_name, frame_length=30):
    """
    CHANGED: This function now accepts the audio data directly, not a file path.
    It also accepts the 'vad' object as an argument.

    Args:
        vad (webrtcvad.Vad): An instance of the VAD.
        waveform_int16 (np.array): The audio data as int16 samples.
        sample_rate (int): The audio sample rate (must be 16000).
        output_dir (str): Directory where speech segments will be saved.
        file_name (str): Original file name.
        frame_length (int): Frame size in milliseconds.
    """

    if sample_rate != 16000:
        print(f"[ERROR] Sample rate must be 16kHz, but got {sample_rate}")
        return

    # CHANGED: No more torchaudio.load() or type conversion, data is ready
    waveform = waveform_int16

    # Webrtcvad supports 10, 20, or 30 ms frames
    frame_size = int(sample_rate * frame_length / 1000)
    if frame_size not in {160, 320, 480}:
        print(f"[ERROR] Invalid frame size for 16kHz: {frame_size} samples")
        return

    segments = []
    current_segment = []

    for i in range(0, len(waveform) - frame_size, frame_size):
        frame = waveform[i: i + frame_size]
        frame_bytes = frame.tobytes()

        # Frame buffer size check (2 bytes per int16 sample)
        if len(frame_bytes) != frame_size * 2:
            continue

        try:
            is_speech = vad.is_speech(frame_bytes, sample_rate)
        except Exception as e:
            # This can happen if the frame is invalid
            # print(f"VAD error: {e}")
            is_speech = False

        if is_speech:
            current_segment.append(frame)
        elif current_segment:
            segments.append(np.concatenate(current_segment))
            current_segment = []

    if current_segment:
        segments.append(np.concatenate(current_segment))

    # Filter segments by length
    min_word_length = 0.2  # 200 ms
    max_word_length = 5.0  # 5 seconds
    sr = sample_rate
    filtered_segments = [seg for seg in segments if min_word_length * sr < len(seg) < max_word_length * sr]

    os.makedirs(output_dir, exist_ok=True)

    heppi = file_name.replace(".wav", "")  # Use os.path.splitext for robustness

    for i, seg in enumerate(filtered_segments):
        # CHANGED: Fixed file name formatting
        output_file = os.path.join(output_dir, f"{heppi}_word_{i}.wav")
        try:
            sf.write(output_file, seg, samplerate=sr)
        except Exception as e:
            print(f"[ERROR] Failed to write {output_file}: {e}")


def process_file_worker(task_tuple):
    """
    NEW: This is the worker function that each parallel process will run.
    It handles preprocessing and segmentation for a single file.
    """
    file_path, output_dir, file_name = task_tuple

    # Each process must have its own VAD object
    vad = webrtcvad.Vad(3)

    # 1. Preprocess in memory
    waveform_int16, sample_rate = preprocess_audio_in_memory(file_path, target_sr=16000)

    if waveform_int16 is None:
        return f"Failed: {file_path}"  # Skip this file

    # 2. Get speech segments
    get_speech_segments(vad, waveform_int16, sample_rate, output_dir, file_name)
    return f"Processed: {file_path}"


def process_and_segment_parallel():
    """
    CHANGED: This function now gathers all file paths and distributes
    the work to a multiprocessing pool.
    """
    directory = r"F:\Datasets\VoxCeleb1\dev_wav"
    output_base_dir = r"F:\Datasets\VoxCeleb1\dev_wav_split"

    tasks = []  # List to hold all (file_path, output_dir, file_name) tuples

    print("[INFO] Gathering all files to process...")
    # Use os.walk for easier directory traversal
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Skip already processed files
            if file.endswith("_processed.wav") or not file.endswith(".wav"):
                continue

            file_path = os.path.join(root, file)

            # Create the corresponding output directory structure
            relative_path = os.path.relpath(root, directory)
            output_dir = os.path.join(output_base_dir, relative_path)

            tasks.append((file_path, output_dir, file))

    print(f"[INFO] Found {len(tasks)} files to process.")
    if not tasks:
        print("[INFO] No files to process.")
        return

    # Get CPU count, leave one free for the system
    n_cores = multiprocessing.cpu_count() - 2
    if n_cores < 1:
        n_cores = 1

    print(f"[INFO] Starting parallel processing with {n_cores} workers...")

    # Create a processing pool and run the tasks
    with multiprocessing.Pool(processes=n_cores) as pool:
        results = list(tqdm(pool.imap_unordered(process_file_worker, tasks), total=len(tasks)))

    print("[INFO] All processing complete.")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

    process_and_segment_parallel()