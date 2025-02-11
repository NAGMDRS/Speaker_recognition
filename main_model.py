import torch, sys, os, tqdm, numpy, soundfile, time
import torch.nn as nn
import torch.nn.functional as F
from helperFiles.tools import tuneThresholdfromScore, ComputeErrorRates, ComputeMinDcf
from helperFiles.losses import AAMsoftmax
from base_model import ECAPA_TDNN

class ECAPAModel(nn.Module):
    def __init__(self,  C, n_class, m, s, lr=0.001, lr_decay=0.97, test_step=1, **kwargs):
        super(ECAPAModel, self).__init__()
        self.speaker_encoder = ECAPA_TDNN(C=C)

        self.speaker_loss = AAMsoftmax(n_class=n_class, m=m, s=s)

        self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=2e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=test_step, gamma=lr_decay)
        num_params = sum(param.numel() for param in self.speaker_encoder.parameters()) / (1024 * 1024)
        print(time.strftime("%m-%d %H:%M:%S") + " Model parameter number = %.2f MB" % num_params)

    def train_network(self, epoch, loader):
        self.train()
        self.scheduler.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        for num, (data, labels) in enumerate(loader, start=1):
            self.zero_grad()
            labels = torch.LongTensor(labels)
            speaker_embedding = self.speaker_encoder.forward(data, aug=True)
            nloss, prec = self.speaker_loss.forward(speaker_embedding, labels)
            nloss.backward()
            self.optim.step()
            index += len(labels)
            top1 += prec
            loss += nloss.detach().cpu().numpy()
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") +
                             " [%2d] Lr: %5f, Training: %.2f%%, Loss: %.5f, ACC: %2.2f%% \r" %
                             (epoch, lr, 100 * (num / len(loader)), loss / num, top1 / index * len(labels)))
            sys.stderr.flush()
        sys.stdout.write("\n")
        return loss / num, lr, top1 / index * len(labels)

    def eval_network(self, eval_list="params/eval_list.txt", eval_path="F:/Datasets/IndianVoxCeleb/vox_indian_split"):
        self.eval()
        files = []
        embeddings = {}
        lines = open(eval_list).read().splitlines()
        for line in lines:
            files.append(line.split()[1])
            files.append(line.split()[2])
        setfiles = sorted(list(set(files)))

        for idx, file in tqdm.tqdm(enumerate(setfiles), total=len(setfiles)):
            audio, _ = soundfile.read(os.path.join(eval_path, file))
            data_1 = torch.FloatTensor(numpy.stack([audio], axis=0))
            max_audio = 300 * 160 + 240
            if audio.shape[0] <= max_audio:
                shortage = max_audio - audio.shape[0]
                audio = numpy.pad(audio, (0, shortage), 'wrap')
            feats = []
            startframe = numpy.linspace(0, audio.shape[0] - max_audio, num=5)
            for asf in startframe:
                feats.append(audio[int(asf):int(asf) + max_audio])
            feats = numpy.stack(feats, axis=0).astype(numpy.float64)
            data_2 = torch.FloatTensor(feats)
            with torch.no_grad():
                embedding_1 = self.speaker_encoder.forward(data_1, aug=False)
                embedding_1 = F.normalize(embedding_1, p=2, dim=1)
                embedding_2 = self.speaker_encoder.forward(data_2, aug=False)
                embedding_2 = F.normalize(embedding_2, p=2, dim=1)
            embeddings[file] = [embedding_1, embedding_2]

        scores, labels = [], []
        for line in lines:
            embedding_11, embedding_12 = embeddings[line.split()[1]]
            embedding_21, embedding_22 = embeddings[line.split()[2]]
            score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T))
            score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
            score = (score_1 + score_2) / 2
            score = score.detach().cpu().numpy()
            scores.append(score)
            labels.append(int(line.split()[0]))

        EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
        fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
        minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
        return EER, minDCF

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model." % origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s" %
                      (origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)

    def extract_embedding(self, x):
        with torch.no_grad():
            embedding = self.speaker_encoder.forward(x, aug=False)
            return F.normalize(embedding, p=2, dim=1)