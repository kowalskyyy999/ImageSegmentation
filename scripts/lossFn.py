#!/usr/bin/python3
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLossMultiClass(nn.Module):
    def __init__(self, encoder, smooth=1):
        super(DiceLossMultiClass, self).__init__()
        self.smooth=smooth
        self.encoder=encoder

    def forward(self, logits, mask):
        N, H, W = logits.size()

        log_temp = logits.new(N, H, W).fill_(-1)
        mask_temp = mask.new(N, H, W).fill_(-1)

        logits = logits.view(-1)
        mask = mask.view(-1)

        log_temp = log_temp.view(-1)
        mask_temp = mask_temp.view(-1)

        multipleLoss = 0
        multipleDiceCoef = 0
        i = 0
        for k, v in self.encoder.items():
            log_temp[logits == v] = 1
            mask_temp[mask == v] = 1
            log_temp[logits != v] = 0
            mask_temp[mask != v] = 0

            coef = self._diceCoef(log_temp, mask_temp, self.smooth)
            loss = 1 - coef

            multipleLoss += loss
            multipleDiceCoef += coef

            i += 1

        multipleLoss = multipleLoss / i
        multipleDiceCoef = multipleDiceCoef / i

        return multipleLoss, multipleDiceCoef 

    def _diceCoef(self, logits, mask, smooth=1):
        intersection = torch.sum(logits * mask)
        return (2 * intersection + smooth) / (torch.sum(logits) + torch.sum(mask) + smooth)

def diceCoef(logits, mask, smooth=1):
    intersection = torch.sum(logits * mask)
    return (2 * intersection + smooth) / (torch.sum(logits) + torch.sum(mask) + smooth)

def getLoss(output, mask):
    ignoreIndex = 17
    entropyLoss = nn.NLLLoss(ignore_index=17)(nn.LogSoftmax(dim=1)(output), mask)
    pred = F.softmax(output, dim=1)
    _, pred = torch.max(pred, 1)
    diceCoef, diceLoss = DiceLossMultiClass()(pred, mask)
    loss = entropyLoss + 1 * diceLoss
    return loss, diceCoef
