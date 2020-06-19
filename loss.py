# loss.py
#
# Author:     Quoc-Huy Nguyen
# Project:    DGCN-torch
# Description:
#             This file contains the dual loss modules.
# 



import torch
import torch.nn as nn



class MaskedMSELoss(nn.Module):
    r"""
    Masked mean square error loss.
    """
    
    def __init__(self, reduction='mean'):
        super(MaskedMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target, target_mask):
        n = target.numel()
        result = ((input - target) ** 2).sum(axis=1)
        result *= target_mask
        if self.reduction == 'mean':
            result = torch.sum(result) / n
        elif self.reduction == 'sum':
            result = torch.sum(result)
            
        return result


class MaskedCrossEntropyLoss(nn.Module):
    r"""
    Masked cross entropy loss.
    """
    
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()
    
    def forward(self, input, target, target_mask):
        #TODO: this is for the nell_dataset
        return input


class DualLoss(nn.Module):
    r"""
    Dual loss.
    """
    
    def __init__(self, supervised_loss_fnc, unsupervised_loss_fnc):
        super(DualLoss, self).__init__()
        self.supervised_loss_fnc = supervised_loss_fnc
        self.unsupervised_loss_fnc = unsupervised_loss_fnc

    def forward(self, a_input, ppmi_input, target, target_mask):
        # TODO: How to combinte the two losses.
        weight = nn.Parameter(torch.rand(1))
        
        result = (self.supervised_loss_fnc(a_input, target, target_mask)
                  + weight * self.unsupervised_loss_fnc(a_input, ppmi_input))
        return result
