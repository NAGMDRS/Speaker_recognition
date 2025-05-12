import torch, math
import torch.nn as nn
import torch.nn.functional as F
from helperFiles.tools import accuracy

class AAMsoftmax(nn.Module):
    """
    Additive Angular Margin Softmax (AAM-Softmax) loss module.

    This is used to enhance class separability by adding an angular margin
    between classes in the feature space, commonly used in speaker verification
    and face recognition tasks.

    Args:
        n_class (int): Number of output classes.
        m (float): Angular margin penalty.
        s (float): Scaling factor for logits.
    """
    def __init__(self, n_class, m, s):
        super(AAMsoftmax, self).__init__()
        self.m = m
        self.s = s
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, 256), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x, label=None):
        """
        Forward pass of the AAM-Softmax layer.

        Args:
            x (Tensor): Input feature tensor of shape (batch_size, 256).
            label (Tensor, optional): Ground truth class labels of shape (batch_size,).

        Returns:
            tuple: Cross-entropy loss and top-1 accuracy percentage.
        """
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        loss = self.ce(output, label)
        prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]
        return loss, prec1