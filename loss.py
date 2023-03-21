import torch
import torch.nn as nn

import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        # 为了防止除0的发生
        smooth = 1

        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score
    

class CE_DiceLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', weight=None):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = SoftDiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction)
    
    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        return CE_loss + dice_loss