# utilities.py
#
# Author:     Quoc-Huy Nguyen
# Project:    DGCN-torch
# Description:
#             This file contains some utility functions.
# 



import torch



def sample_mask(indices, length):
    r"""
    Creates a boolean mask which is a torch.BoolTensor with specified 
    integer length and one's at the specified indices.
    
    Parameters
    ----------
    indices : LongTensor
        The indices to activate in the mask.
    length : int
        Lenth of the output boolean mask.
        
    Returns
    -------
    mask : torch.BoolTensor
        The boolean mask.
    """
    
    mask = torch.zeros(length, dtype=torch.bool)
    mask[indices] = 1
    return mask



def yield_batch_mask(mask, batch_size):
    r"""
    Yields a new boolean mask containing at most a batch_size number of one's
    from the original mask, but randomly permuted.

    Parameters
    ----------
    mask : torch.BoolTensor
        The input binary mask.
    batch_size : int
        Batch size.

    Yields
    -------
    batch_mask : torch.BoolTensor
        The output binary mask.
    """
    
    if batch_size == 0:
        yield mask
    else:
        indices = torch.nonzero(mask)
        sample_size = indices.shape[0]
        mask_size = mask.shape[0]
        permuted_indices = indices[torch.randperm(sample_size)]
        for i in range(0, sample_size, batch_size):
            batch_mask = sample_mask(permuted_indices[i : i + batch_size], mask_size)
            yield batch_mask



def accuracy(input, target, target_mask):
    r"""
    Computes an accuracy value for the prediction labels of `input` compared to the
    true labels of `target` according to the boolean mask `target_mask` which
    is a torch.BoolTensor.
    
    Parameters
    ----------
    input : torch.Tensor
        The input tensor of the prediction labels.
    target : torch.Tensor
        The target tensor of the true labels.
    target_mask : torch.BoolTensor
        The boolean mask for the actual target examples.
    
    Returns
    -------
    float
        Accuracy score between 0.0 and 1.0.
    """
    
    y_preds = torch.argmax(input, axis=1)
    y_labels = torch.argmax(target, axis=1)
    all_compares = y_preds == y_labels
    all_compares *= target_mask
    return torch.sum(all_compares) / torch.sum(target_mask, dtype=torch.float32)
    

    