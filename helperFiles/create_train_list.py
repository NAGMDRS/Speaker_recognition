import os


def create_list_fast(root_dir,
                     output_file=r"C:\Users\Akshay Gupta\PycharmProjects\speaker_recognition\params\train_list(dev).txt"):
    entry_count = 0

    with open(output_file, "w") as f:
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, root_dir)
                    speaker_id = relative_path.split(os.sep)[0]
                    f.write(f"{speaker_id} {relative_path.replace(os.sep, '/')}\n")
                    entry_count += 1

    print(f"✅ train_list.txt created with {entry_count} entries!")


create_list_fast(r"F:\Datasets\VoxCeleb1\dev_wav_split")
