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
    
    def __init__(self, in_features : int, out_features : int,
                 diffusion : torch.tensor, dropout_rate : float = 0.3,
                 W : torch.tensor = None, activation=nn.ReLU(inplace=True)) -> None:
        '''
        Hidden dense layer as defined in DGCN paper.

        Parameters
        ----------
        in_features : int
            Number of features of input tensor.
        out_features : int
            Number of features of output tensor.
        diffusion : torch.tensor
            Diffusion matrix, i.e. the adjacency matrix or positive pointwise
            mutual information matrix.
        dropout_rate : float
            Dropout rate for randomly settings some elements as zero. 
        W : torch.tensor, optional
            User-defined weight matrix. The default is None.
        activation : optional
            An activation layer from torch.nn. The default is 
            nn.ReLU(inplace=True).

        Returns
        -------
        None.
        '''
        
        super(hidden_dense_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.diffusion = diffusion
        self.layers = nn.Sequential()
        
        linear = nn.Linear(in_features, out_features, bias=False)
        
        # reassign weight to linear layer if given one
        if W is not None:
            # if type(W) != torch.tensor:
            #     raise TypeError(f"hidden_dense_layer.__init__: W must be type torch.tensor, got ({type(W)})")
            if W.size() != (out_features, in_features):
                raise ValueError(f"hidden_dense_layer.__init__: W must be size {(out_features, in_features)}, got ({W.size()})")
            linear.weight.data = W
        
        self.weight = linear.weight
        
        if dropout_rate != 0.0:
            dropout = nn.Dropout(dropout_rate, inplace=True)
            self.layers.add_module("dropout", dropout)
            
        self.layers.add_module("linear", linear)
        self.layers.add_module("activation", activation)
        
    
    def forward(self, input : torch.tensor) -> torch.tensor:
        '''
        Pass through the hidden layer.

        Parameters
        ----------
        input : torch.tensor
            Input tensor.

        Returns
        -------
        torch.tensor
            Output tensor.
        '''
        return self.layers(self.diffusion.mm(input))
    
