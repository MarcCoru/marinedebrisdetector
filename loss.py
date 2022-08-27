from torch import nn
import torch
def get_loss(pos_weight=None):
    bcecriterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    def criterion(y_pred, target, mask=None):
        """a wrapper around BCEWithLogitsLoss that ignores no-data
        mask provides a boolean mask on valid data"""
        loss = bcecriterion(y_pred, target)
        if mask is not None:
            return (loss * mask.double()).mean()
        else:
            return loss.mean()
    return criterion
