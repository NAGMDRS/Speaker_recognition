import torch, sys, tqdm, numpy, soundfile, time
import torch.nn as nn
import torch.nn.functional as F
from helperFiles.tools import tuneThresholdfromScore, ComputeErrorRates, ComputeMinDcf
from helperFiles.losses import AAMsoftmax
from base_model import ECAPA_TDNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class ECAPAModel(nn.Module):
    def __init__(self, C, n_class, m, s, lr=0.001, lr_decay=0.97, test_step=1, **kwargs):
        super(ECAPAModel, self).__init__()
        self.speaker_encoder = ECAPA_TDNN(C=C).to(device)
        self.speaker_loss = AAMsoftmax(n_class=n_class, m=m, s=s).to(device)
        self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=2e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=test_step, gamma=lr_decay)
        num_params = sum(param.numel() for param in self.speaker_encoder.parameters()) / (1024 * 1024)
        print(time.strftime("%m-%d %H:%M:%S") + f" Model parameter number = {num_params:.2f} MB")

    def train_network(self, epoch, loader):
        self.train()
        self.scheduler.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        for num, (data, labels) in enumerate(loader, start=1):
            self.zero_grad()
            data, labels = data.to(device), labels.to(torch.long).to(device)
            speaker_embedding = self.speaker_encoder.forward(data, aug=True)
            nloss, prec = self.speaker_loss.forward(speaker_embedding, labels)
            nloss.backward()
            self.optim.step()
            index += len(labels)
            top1 += prec
            loss += nloss.detach().cpu().numpy()
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") +
                             f" [{epoch:2d}] Lr: {lr:.6f}, Training: {100 * (num / len(loader)):.2f}%, "
                             f"Loss: {float(loss / num):.5f}, ACC: {float(top1 / index * len(labels)):.2f}% \r")
            sys.stderr.flush()

        sys.stdout.write("\n")
        return loss / num, lr, top1 / index * len(labels)

    def eval_network(self, eval_list="/home/btech10154.22/Speaker_recognition-main/params/train_list.txt",
                     eval_path="/home/btech10154.22/vox_indian_split"):
        self.eval()
        files, embeddings = [], {}
        lines = open(eval_list).read().splitlines()
        for line in lines:
            if len(line.split()) == 3:
                files.extend([line.split()[1], line.split()[2]])
            else:
                print(line)

        setfiles = sorted(set(files))

        for idx, file in tqdm.tqdm(enumerate(setfiles), total=len(setfiles)):
            audio, _ = soundfile.read(f"{eval_path}/{file}")
            data_1 = torch.FloatTensor(numpy.stack([audio], axis=0)).to(device)
            max_audio = 300 * 160 + 240
            if audio.shape[0] <= max_audio:
                audio = numpy.pad(audio, (0, max_audio - audio.shape[0]), 'wrap')
            feats = numpy.stack(
                [audio[int(asf):int(asf) + max_audio] for asf in numpy.linspace(0, audio.shape[0] - max_audio, num=5)],
                axis=0)
            data_2 = torch.FloatTensor(feats).to(device)
            with torch.no_grad():
                embedding_1 = F.normalize(self.speaker_encoder.forward(data_1, aug=False), p=2, dim=1)
                embedding_2 = F.normalize(self.speaker_encoder.forward(data_2, aug=False), p=2, dim=1)
            embeddings[file] = [embedding_1, embedding_2]

        scores, labels = [], []
        for line in lines:
            embedding_11, embedding_12 = embeddings[line.split()[1]]
            embedding_21, embedding_22 = embeddings[line.split()[2]]
            score = (torch.mean(torch.matmul(embedding_11, embedding_21.T)) + torch.mean(
                torch.matmul(embedding_12, embedding_22.T))) / 2
            scores.append(score.detach().cpu().numpy())
            labels.append(int(line.split()[0]))

        EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
        fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
        minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
        return EER, minDCF

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path, map_location=device)
        for name, param in loaded_state.items():
            origname = name.replace("module.", "") if name not in self_state else name
            if origname in self_state and self_state[origname].size() == param.size():
                self_state[origname].copy_(param)
            else:
                print(f"Skipping {origname}: shape mismatch or missing in model.")

    def extract_embedding(self, x):
        with torch.no_grad():
            return F.normalize(self.speaker_encoder.forward(x.to(device), aug=False), p=2, dim=1)

    def forward(self, x):
        with torch.no_grad():
            return self.speaker_encoder.forward(x.to(device), aug=False)
