import torch
import torch.nn as nn


class RMSELoss(nn.Module):

    def __init__(self, reduction='mean'):

        super(RMSELoss, self).__init__()

        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(self, y_true, y_pred):

        loss = torch.sqrt(self.mse_loss(y_pred, y_true))
        return loss
