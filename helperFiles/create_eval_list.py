import random

train_txt_path = "C:/Users/Akshay Gupta/PycharmProjects/speaker_recognition/params/train_list(dev).txt"
eval_list_path = "C:/Users/Akshay Gupta/PycharmProjects/speaker_recognition/params/eval_list(dev).txt"

speaker_files = {}
with open(train_txt_path, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        speaker_id, file_path = parts[0], parts[1]

        if speaker_id not in speaker_files:
            speaker_files[speaker_id] = []
        speaker_files[speaker_id].append(file_path)

eval_pairs = []

for speaker, files in speaker_files.items():
    if len(files) > 1:
        for _ in range(min(5, len(files) - 1)):
            file1, file2 = random.sample(files, 2)
            eval_pairs.append(f"1 {file1} {file2}")

speakers = list(speaker_files.keys())
for _ in range(len(eval_pairs)):
    spk1, spk2 = random.sample(speakers, 2)
    file1 = random.choice(speaker_files[spk1])
    file2 = random.choice(speaker_files[spk2])
    eval_pairs.append(f"0 {file1} {file2}")

random.shuffle(eval_pairs)
with open(eval_list_path, "w") as f:
    f.write("\n".join(eval_pairs))

print(f"Evaluation list saved to {eval_list_path}")
