import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid
from torch import reshape

# PyTorch
ALPHA = 0.5
BETA = 0.5
GAMMA = 1


class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA, gamma=GAMMA):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = reshape(inputs, (-1,))
        targets = reshape(targets, (-1,))

        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        FocalTversky = (1 - Tversky) ** gamma

        return FocalTversky
