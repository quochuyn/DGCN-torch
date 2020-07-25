# layers.py
#
# Author:     Quoc-Huy Nguyen
# Project:    DGCN-torch
# Description:
#             This file contains classes for hidden neural network layers.
# 



import torch
import torch.nn as nn



class hidden_dense_layer(nn.Module):
    r"""
    The hidden dense layer as defined in the DGCN paper combines an unbiased
    linear layer (torch.nn.Linear) followed by an optional batch normalization
    layer (nn.BatchNorm1d), then an activation layer which is, by default, ReLU
    (torch.nn.ReLU) and an optional dropout layer (torch.nn.Dropout).

    Parameters
    ----------
    in_features : int
        Number of features of input tensor.
    out_features : int
        Number of features of output tensor.
    diffusion : torch.Tensor, optional
        Diffusion matrix, i.e. the adjacency matrix or positive pointwise
        mutual information matrix. The default is None.
    dropout_rate : float, optional
        Dropout rate for randomly settings some elements as zero. The default
        is 0.3.
    W : torch.Tensor, optional
        User-defined weight matrix. The default is None.
    batch_norm : bool, optional
        Boolean value whether to include a batch normalization layer after
        the linear pass.
    activation : str, optional
        An activation layer from torch.nn. The default is 'relu'.
    """
    
    
    def __init__(self, in_features : int, out_features : int,
                 diffusion : torch.Tensor = None, dropout_rate : float = 0.3,
                 W : torch.Tensor = None, batch_norm = False,
                 activation = 'relu'):
        
        super(hidden_dense_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.diffusion = diffusion
        self.layers = nn.Sequential()
        
        linear = nn.Linear(in_features, out_features, bias=False)
        activation = get_activation(activation)
        
        # reassign weight to linear layer if given one
        if W is not None:
            if W.size() != (out_features, in_features):
                raise ValueError(f"hidden_dense_layer.__init__: W must be size {(out_features, in_features)}, got ({W.size()})")
            linear.weight.data = W
        
        # give access to the linear weight parameters if need to share
        self.weight = linear.weight
            
        self.layers.add_module('linear', linear)
        if batch_norm == True:
            self.layers.add_module('batchnorm', nn.BatchNorm1d(out_features))
        self.layers.add_module('activation', activation)
        self.layers.add_module('dropout', nn.Dropout(dropout_rate))
        
        
    def forward(self, input : torch.Tensor) -> torch.Tensor:
        if self.diffusion is not None:
            return self.layers(self.diffusion.matmul(input))
        else:
            return self.layers(input)


def get_activation(activation):
    if activation == 'relu':
        activation = nn.ReLU()
    elif activation == 'softmax':
        activation = nn.Softmax(dim=1)
    elif activation == 'tanh':
        activation = nn.Tanh()
    elif activation == 'sigmoid':
        activation = nn.Sigmoid()
    else:
        assert False, f"get_activation: activation function '{activation}' not supported"

    return activation


