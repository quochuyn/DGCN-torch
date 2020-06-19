# utilities.py
#
# Author:     Quoc-Huy Nguyen
# Project:    DGCN-torch
# Description:
#             This file contains some utility functions.
# 



import torch



def _sample_mask(indices, length):
    r"""
    Create a template boolean mask which is a torch.BoolTensor with specified 
    integer length and one's at the specified indices.
    """
    
    mask = torch.zeros(length, dtype=torch.bool)
    mask[indices] = 1
    return mask


def accuracy(input, target, target_mask):
    r"""
    Computes an accuracy value for the prediction labels of `input` compared to the
    true labels of `target` according to the boolean mask `target_mask` which
    is a torch.BoolTensor.
    """
    
    y_preds = torch.argmax(input, axis=1)
    y_labels = torch.argmax(target, axis=1)
    all_compares = y_preds == y_labels
    all_compares *= target_mask
    return torch.sum(all_compares) / torch.sum(target_mask, dtype=torch.float32)


