# model.py
#
# Author:     Quoc-Huy Nguyen
# Project:    DGCN-torch
# Description:
#             This is a PyTorch implementation of DGCN, a Dual Graph 
#             Convolutional Neural Network for graph-based semi-supervised 
#             classification described in the Chenyi Zhuang, Qiang Ma paper.
# 



import torch
import torch.nn as nn

from layers import hidden_dense_layer



class DGCN(nn.Module):
    
    def __init__(self, layer_sizes : [(int, int)], adjacency : torch.tensor, 
                 ppmi : torch.tensor, dropout_rate : float = 0.3, 
                 activation = None) -> None:
        '''
        Dual Graph Convolutional Neural Network.

        Parameters
        ----------
        layer_sizes : [(int, int)]
            List of 2-tuples for hidden layer sizes.
        adjacency : torch.tensor
            Adjacency matrix of the graph.
        ppmi : torch.tensor
            Positive pointwise mutual information matrix of the graph.
        activation : optional
            An activation layer from torch.nn. The default is None.

        Returns
        -------
        None
        '''
        
        self.a_layers = nn.Sequential()
        self.ppmi_layers = nn.Sequential()
        
        # define the dual NN sharing the same weight W
        for index, (n_in, n_out) in enumerate(layer_sizes):
            _hidden_layer_a = hidden_dense_layer(
                in_featuers  = n_in,
                out_featuers = n_out,
                diffusion    = ppmi,
                dropout_rate = (0.0 if index == 0 else dropout_rate))
            self.a_layers.add_module(f'hidden_a{index}', _hidden_layer_a)
            
            _hidden_layer_ppmi = hidden_dense_layer(
                in_featuers  = n_in,
                out_featuers = n_out,
                diffusion    = ppmi,
                W            = _hidden_layer_a.weight,
                dropout_rate = (0.0 if index == 0 else dropout_rate))
            self.a_layers.add_module(f'hidden_ppmi{index}', _hidden_layer_ppmi)

        
        self.a_layers.add_module('final_activation_a', activation)
        self.ppmi_layers.add_module('final_activation_ppmi', activation)
        
        
        
        
    
    
    def forward(self, x : torch.tensor) -> torch.tensor:
        '''
        Pass through the DGCN.

        Parameters
        ----------
        x : torch.tensor
            Input tensor.

        Returns
        -------
        torch.tensor
            Output tensor.
        '''
        
        return x
    
    
    