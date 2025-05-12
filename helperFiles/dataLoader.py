import numpy, random, soundfile, torch, os
import config as cfg  # Import the configuration
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class train_loader(object):  
    def __init__(self, num_frames=cfg.NUM_FRAMES,   
                 train_path=cfg.DATA_PATH,  
                 train_list=cfg.TRAIN_LIST,  
                 segment_audio=cfg.SEGMENT_AUDIO, **kwargs):  
          
        self.train_path = train_path  
        self.num_frames = num_frames  
          
        # Loading Data and Labels  
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
        # Read the utterance and randomly select the segment  
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
          
        return torch.FloatTensor(audio[0]).to(cfg.DEVICE), torch.tensor(self.data_label[index], dtype=torch.long).to(cfg.DEVICE)  
      


def __len__(self):
        return len(self.data_list)
