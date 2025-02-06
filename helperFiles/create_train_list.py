import os


def create_list(root_dir, output_file="params/train_list.txt"):
    with open(output_file, "w") as f:
        for speaker_id in os.listdir(root_dir):
            speaker_path = os.path.join(root_dir, speaker_id)
            if os.path.isdir(speaker_path):
                for root, _, files in os.walk(speaker_path):
                    for file in files:
                        if file.endswith('.wav'):
                            file_path = os.path.join(root, file)
                            relative_path = os.path.relpath(file_path, root_dir)
                            f.write(f"{speaker_id} {relative_path}\n")

    print(f"✅ train_list.txt created with {sum(1 for _ in open(output_file))} entries!")


create_list(r"F:\Datasets\IndianVoxCeleb\vox_indian_split")
