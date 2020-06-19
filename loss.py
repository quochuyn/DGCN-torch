# loss.py
#
# Author:     Quoc-Huy Nguyen
# Project:    DGCN-torch
# Description:
#             This file contains loss functions.
# 



import torch
import torch.nn as nn



class MaskedMSELoss(nn.Module):
    r"""
    Creates a criterion that measures the mean squared error (squared 
    L2 norm) between each element in the input `x` and target `y` according
    to the boolean mask `target_mask` which is a BoolTensor.
    
    The unreduced loss (i.e. when reduction is 'none') can be described as
    
    .. math::
        
        \ell(x,y) = L = \{l_1,\dots,l_N}^\top * M, \quad
        l_n = \left( x_n - y_n \right)^2,
        
    where `x` and `y` are tensors of arbitrary shape with `n` elements, `N` 
    is the batch size, `*` the element-wise multiplication operator, and `M`
    the boolean mask.

    If not 'none', reduction is 'mean' by default which operates over all the 
    elements and divides by the total number of elements `n`.
    
    The division by `n` can be avoided, if reduction is 'sum'.
    
    Parameters
    ----------
    reduction : str, optional
        Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        'none': no reduction will be applied. 'mean': the sum of the output
        will be divided by the number of elements in the output. 'sum': the
        output will be summed. The default is 'mean'.
    """
    
    def __init__(self, reduction='mean'):
        super(MaskedMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target, target_mask):
        result = ((input - target) ** 2).sum(axis=1)
        result *= target_mask
        if self.reduction != 'none':
            if self.reduction == 'mean':
                result = torch.mean(result)
            else:
                result = torch.sum(result)
            
        return result


class MaskedCrossEntropyLoss(nn.Module):
    r"""
    Masked cross entropy loss. TODO
    """
    
    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super(MaskedCrossEntropyLoss, self).__init__()
    
    def forward(self, input, target, target_mask):
        #TODO: this is for the nell_dataset
        return input


class DualLoss(nn.Module):
    r"""
    Dual loss combining both supervised and unsupervised loss. TODO
    """
    
    def __init__(self, supervised_loss_fnc, unsupervised_loss_fnc):
        super(DualLoss, self).__init__()
        self.supervised_loss_fnc = supervised_loss_fnc
        self.unsupervised_loss_fnc = unsupervised_loss_fnc

    def forward(self, a_input, ppmi_input, target, target_mask):
        weight = nn.Parameter(torch.rand(1))
        
        result = (self.supervised_loss_fnc(a_input, target, target_mask)
                  + weight * self.unsupervised_loss_fnc(a_input, ppmi_input))
        return result



