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
        
        \ell(x,y) = L = \{l_1,\dots,l_N}^\top * \{m_1,\dots,m_N\}, \quad
        l_n = (x_n - y_n)^2, \quad m_n \in \{0,1\}
        
    where `x` and `y` are tensors of arbitrary shape with `n` elements, `N` 
    is the batch size, `*` the element-wise multiplication operator, and 
    `\{m_1,\dots,m_N\}` the boolean mask.

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
    Creates a criterion that measures the cross entropy loss between each
    element in the input `x` and target `y` according to the boolean mask 
    `target_mask` which is a BoolTensor.
    
    The unreduced loss (i.e. when reduction is 'none') can be described as
    
        \ell(x,y) = L = \{l_1,\dots,l_N}^\top * \{m_1,\dots,m_N\}, \quad 
        l_n = - (y_n * \log(x_n)), \quad m_n \in \{0,1\} 
        
    where `x` and `y` are tensors of arbitrary shape with `n` elements, `N` 
    is the batch size, `*` the element-wise multiplication operator, and 
    `\{m_1,\dots,m_N\}` the boolean mask.
    
    If not 'none', reduction is 'mean' by default which operates over all the 
    elements and divides by the total number of elements `n`.
    
    The division by `n` can be avoided, if reduction is 'sum'.
    
    Parameters
    ----------
    eps : float, optional
        Small value to avoid evaluation of \log(0). The default is 1e-8.
    reduction : str, optional
        Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        'none': no reduction will be applied. 'mean': the sum of the output
        will be divided by the number of elements in the output. 'sum': the
        output will be summed. The default is 'mean'.
    """
    
    def __init__(self, eps=1e-8, reduction='mean'):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.reduction = reduction
        self.eps = eps
        
    def forward(self, input, target, target_mask):
        result = -(target * (input + self.eps).log()).sum(axis=1)
        result *= target_mask
        if self.reduction != 'none':
            if self.reduction == 'mean':
                result = torch.mean(result)
            else:
                result = torch.sum(result)
        return result
        


class DualLoss(nn.Module):
    r"""
    The dual loss combines both a supervised and an unsupervised loss in one
    single class. The learnable weight that combines the two losses is 
    randomly initialized from a standard normal distribution scaled down. 
    
    Parameters
    ----------
    supervised_loss_fnc : torch.nn.Module
        Supervised loss with respect to the labeled data.
    unsupervised_loss_fnc : torch.nn.Module
        Unsupervised loss with respect to the graph structure.
    """
    
    def __init__(self, supervised_loss_fnc, unsupervised_loss_fnc):
        super(DualLoss, self).__init__()
        self.supervised_loss_fnc = supervised_loss_fnc
        self.unsupervised_loss_fnc = unsupervised_loss_fnc
        
    def forward(self, a_input, ppmi_input, target, target_mask, weight=None):
        if weight == None:
            weight = 1.0
        supervised_loss = self.supervised_loss_fnc(a_input, target, target_mask)
        unsupervised_loss = self.unsupervised_loss_fnc(a_input, ppmi_input)
        result = (supervised_loss + weight * unsupervised_loss)
        # print(f"  supervised_loss: {supervised_loss}")
        # print(f"  unsupervised_loss: {unsupervised_loss}")
        return result


