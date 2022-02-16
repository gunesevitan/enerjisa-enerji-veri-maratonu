import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSELoss(nn.Module):

    def __init__(self, reduction='mean'):

        super(RMSELoss, self).__init__()

        self.reduction = reduction

    def forward(self, inputs, targets):

        return torch.sqrt(F.mse_loss(inputs, targets, reduction=self.reduction))
