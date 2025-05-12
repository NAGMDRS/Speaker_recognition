import numpy, random, soundfile, torch, os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class train_loader(object):
    """
    Data loader for training speaker recognition models.

    This loader reads a list of audio file paths and corresponding speaker labels,
    processes each audio sample by extracting a fixed-length segment, and returns
    the audio tensor along with its label.

    Args:
        num_frames (int): Number of frames per training segment (default: 200).
        train_path (str): Path to the directory containing training audio files.
        train_list (str): Path to the text file containing speaker ID and relative audio paths.
        segment_audio (bool): Unused flag for future segmenting logic.
        **kwargs: Additional arguments for compatibility.
    """
    def __init__(self, num_frames=200, train_path="/home/btech10154.22/vox_indian_split",
                 train_list="/home/btech10154.22/Speaker_recognition-main/params/train_list.txt",
                 segment_audio=False, **kwargs):

        self.train_path = train_path
        self.num_frames = num_frames

        self.data_list = []
        self.data_label = []

        with open(train_list, "r") as f:
            lines = f.read().splitlines()

        dictkeys = list(set([x.split()[0] for x in lines]))
        dictkeys.sort()
        dictkeys = {key: val for val, key in enumerate(dictkeys)}
        print(dictkeys)

        for index, line in enumerate(lines):
            speaker_label = dictkeys[line.split()[0]]
            file_name = os.path.join(train_path, line.split()[1]).replace("\\", "/")  # Fix backslashes
            self.data_label.append(speaker_label)
            self.data_list.append(file_name)

    def __getitem__(self, index):
        """
        Retrieves a processed audio segment and its corresponding label.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - torch.FloatTensor: The audio waveform segment.
                - torch.LongTensor: The speaker label.
        """
        file_path = self.data_list[index]

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        audio, sr = soundfile.read(file_path)
        length = self.num_frames * 160 + 240

        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), 'wrap')

        start_frame = numpy.int64(random.random() * (audio.shape[0] - length))
        audio = audio[start_frame:start_frame + length]
        audio = numpy.stack([audio], axis=0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ...
        return torch.FloatTensor(audio[0]).to(device), torch.tensor(self.data_label[index], dtype=torch.long).to(device)


    def __len__(self):
        return len(self.data_list)
