import torch
import torch.nn as nn
from torch.nn import functional as F
import logging
from config import config


class CrossEntropy(nn.Module):
    def __init__(self, ignore_index=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index
        )

    def _forward(self, score, target):
        loss = self.criterion(score, target)
        return loss

    def forward(self, score, target):
        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]
        weights = config.LOSS.BALANCE_WEIGHTS
        assert len(weights) == len(score)
        return sum([w * self._forward(x, target) for (w, x) in zip(weights, score)])

"""
    Dice Loss
    Dice系数：语义分割的常用评价指标之一
"""
class SoftDice(nn.Module):
    def __init__(self, num_classes, activation='sigmoid', reduction='mean'):
        super(SoftDice, self).__init__()
        self.activation = activation
        self.num_classes = num_classes
    
    def _diceCoeff(self, score, target, num_classes, smooth=1e-5, activation='sigmoid'):
        """ computational formula：
            dice = (2 * (score ∩ target)) / (score ∪ target)
        """
        device = score.device
        if activation is None or activation == "none":
            activation_fn = lambda x: x
        elif activation == "sigmoid":
            activation_fn = nn.Sigmoid()
        elif activation == "softmax2d":
            activation_fn = nn.Softmax2d()
        else:
            raise NotImplementedError("Activation implemented for sigmoid and softmax2d 激活函数的操作")
        score = activation_fn(score).to(device)
        N = target.size(0) # 除以batch数
        score_flat = score.view(N, -1).to(device)
        target_flat = target.type(torch.FloatTensor).view(N, -1).to(device)
    
        intersection = (score_flat * target_flat).sum(1).to(device)
        unionset = score_flat.sum(1) + target_flat.sum(1).to(device)
        loss = (2 * intersection + smooth) / (unionset + smooth)
        return loss.sum() / N # batch平均损失

    def forward(self, scores, targets):
        class_dice = []
        device = scores.device
        for i in range(self.num_classes):
            class_dice.append(self._diceCoeff(
                scores[:, i:i + 1, :].to(device), 
                targets == torch.full(targets.size(), i * 1.0).to(device), 
                num_classes=self.num_classes, 
                activation=self.activation
            ))
        mean_dice = sum(class_dice) / len(class_dice)
        return 1 - mean_dice

class Combo(nn.Module):
    def __init__(self, num_classes, ignore_index=-1, weight=None):
        super(Combo, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = []
        self.criterion.append(nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index
        ))
        self.criterion.append(SoftDice(
            num_classes=num_classes
        ))
        
    def _forward(self, score, target, loss_id):
        loss = self.criterion[loss_id](score, target)
        return loss

    def forward(self, score, target):
        weights = config.LOSS.BALANCE_WEIGHTS
        assert len(weights) == len(self.criterion)
        return sum([w * self._forward(score, target, loss_id) for loss_id,w in enumerate(weights)])