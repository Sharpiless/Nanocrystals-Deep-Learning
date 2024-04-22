import torch.nn.init as init
import torch.nn as nn
import logging

import torch
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2., reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=-1)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        probs = torch.exp(log_probs)
        loss = -((1 - probs) ** self.gamma) * log_probs * targets
        if self.weight is not None:
            loss = loss * self.weight.unsqueeze(0)

        loss = loss.sum(dim=1)

        if self.ignore_index >= 0:
            non_ignored = targets.sum(1).bool()
            loss = loss[non_ignored]

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# 自定义初始化函数
def init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            init.constant_(m.bias, 0)


def get_logger(save_path):
    logger = logging.getLogger("debug")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s")

    file_handler = logging.FileHandler(save_path)
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger