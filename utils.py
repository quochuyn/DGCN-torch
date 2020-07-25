# utilities.py
#
# Author:     Quoc-Huy Nguyen
# Project:    DGCN-torch
# Description:
#             This file contains some utility functions.
# 



import torch
import math



def get_device():
    r"""
    Get the device to run the computations on (i.e. the gpu or cpu).
    
    Returns
    -------
    device : torch.device
        The device to run the computations on (i.e. the gpu or cpu).
    """
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device



def sample_mask(indices, length, device=None):
    r"""
    Creates a boolean mask which is a torch.BoolTensor with specified 
    integer length and one's at the specified indices.
    
    Parameters
    ----------
    indices : LongTensor
        The indices to activate in the mask.
    length : int
        Lenth of the output boolean mask.
    device : torch.device, optional
        The device to run the computations on (i.e. the gpu or cpu). 
        Default: If None, then uses the current device for the default
        tensor type (see torch.set_default_tensor_type()).
        
    Returns
    -------
    mask : torch.BoolTensor
        The boolean mask.
    """
    
    mask = torch.zeros(length, dtype=torch.bool, device=device)
    mask[indices] = 1
    return mask



def yield_batch_mask(mask, batch_size, device=None):
    r"""
    Yields a new boolean mask containing at most a batch_size number of one's
    from the original mask, but randomly permuted.

    Parameters
    ----------
    mask : torch.BoolTensor
        The input binary mask.
    batch_size : int
        Batch size.
    device : torch.device, optional
        The device to run the computations on (i.e. the gpu or cpu). 
        Default: If None, then uses the current device for the default
        tensor type (see torch.set_default_tensor_type()).

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
            batch_mask = sample_mask(permuted_indices[i : i + batch_size], mask_size, device)
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



def rampup(epoch, scaled_unsup_weight_max, exp=5.0, rampup_length=80):
    if epoch < rampup_length:
        p = max(0.0, float(epoch)) / float(rampup_length)
        p = 1.0 - p
        return math.exp(-p * p * exp) * scaled_unsup_weight_max
    else:
        return 1.0 * scaled_unsup_weight_max



def get_scaled_unsup_weight_max(num_labels, X_train_shape, unsup_weight_max=100.0):
    return unsup_weight_max * 1.0 * num_labels / X_train_shape