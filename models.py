# models.py
#
# Author:     Quoc-Huy Nguyen
# Project:    DGCN-torch
# Description:
#             This is a PyTorch implementation of DGCN, a Dual Graph 
#             Convolutional Neural Network for graph-based semi-supervised 
#             classification described in the Chenyi Zhuang and Qiang Ma paper.
# 



import torch
import torch.nn as nn

import loss
from layers import hidden_dense_layer




class DGCN(nn.Module):
    """
    Dual Graph Convolutional Neural Network. TODO: Better description.

    Parameters
    ----------
    layer_sizes : [(int, int)]
        List of 2-tuples for hidden layer sizes.
    adjacency : torch.tensor
        Adjacency matrix of the graph.
    ppmi : torch.tensor
        Positive pointwise mutual information matrix of the graph.
    dropout_rate : float, optional
        Dropout rate at each layer. The default is 0.3.
    activation : optional
        The final activation layer. The default is torch.nn.Softmax(dim=1).
    nell_dataset : bool
        True if we are testing the nell dataset and False otherwise. The 
        default is false.
    """
    
    def __init__(self, layer_sizes, adjacency, ppmi, dropout_rate = 0.3, 
                 activation = nn.Softmax(dim=1), nell_dataset = False):
        super(DGCN, self).__init__()
        self.a_layers = nn.Sequential()
        self.ppmi_layers = nn.Sequential()
        self.l1 = 0.0
        self.l2 = 0.0
        self.loss = 0.0
        
        # define the dual NN sharing the same weight W
        for index, (n_in, n_out) in enumerate(layer_sizes):
            _hidden_layer_a = hidden_dense_layer(
                in_features  = n_in,
                out_features = n_out,
                diffusion    = adjacency,
                dropout_rate = (0.0 if index == 0 else dropout_rate))
            self.a_layers.add_module(f'hidden_a{index + 1}', _hidden_layer_a)
            
            _hidden_layer_ppmi = hidden_dense_layer(
                in_features  = n_in,
                out_features = n_out,
                diffusion    = ppmi,
                W            = _hidden_layer_a.weight,
                dropout_rate = (0.0 if index == 0 else dropout_rate))
            self.ppmi_layers.add_module(f'hidden_ppmi{index + 1}', _hidden_layer_ppmi)

        # add the final activation layer to apply labels
        self.a_layers.add_module('final_activation_a', activation)
        self.ppmi_layers.add_module('final_activation_ppmi', activation)
        
        # define the supervised loss
        if nell_dataset:
            self.supervised_loss = loss.MaskedCrossEntropyLoss()
        else:
            self.supervised_loss = loss.MaskedMSELoss(reduction='mean')
        
        # define the unsupervised loss
        self.unsupervised_loss = nn.MSELoss(reduction='mean')
        
        # define the dual loss
        self.dual_loss = loss.DualLoss(self.supervised_loss, self.unsupervised_loss)
        
            
        
        
        
    def forward(self, input : torch.tensor) -> torch.tensor:
        output_a = self.a_layers(input)
        output_ppmi = self.ppmi_layers(input)
        
        # define the regularizer #TODO: why do we need this again?
        
        return (output_a, output_ppmi)
    
    
    