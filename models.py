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
    The Dual Graph Convolutional Neural Network (DGCN) is a model for 
    graph-based semi-supervised classifation. It combines two neural
    networks that simultaneously learns the `local consistency-based knowledge' 
    and the `global consistency-based knowledge' of the graph structured data. 
    Respectively, the adjacency and positive pointwise mutual information 
    (PPMI) matrices capture these knowledges.

    Parameters
    ----------
    layer_sizes : [(int, int)]
        List of 2-tuples for hidden layer sizes.
    adjacency : torch.Tensor
        Adjacency matrix of the graph.
    ppmi : torch.Tensor
        Positive pointwise mutual information matrix of the graph.
    dropout_rate : float, optional
        Dropout rate at each layer. The default is 0.3.
    activation : optional
        The activation function for the hidden layers. The default is 
        torch.nn.ReLU().
    final_activation : optional
        The final activation function for the output layer. The default is 
        torch.nn.Softmax(dim=1).
    """
    
    
    def __init__(self, layer_sizes, adjacency, ppmi, dropout_rate = 0.3, 
                 activation = nn.ReLU(), final_activation = nn.Softmax(dim=1)):
        super(DGCN, self).__init__()
        self.a_layers = nn.Sequential()
        self.ppmi_layers = nn.Sequential()
        
        num_layers = len(layer_sizes)
        
        # define the dual NN sharing the same weight W
        for index, (n_in, n_out) in enumerate(layer_sizes, 1):
            _hidden_layer_a = hidden_dense_layer(
                in_features  = n_in,
                out_features = n_out,
                diffusion    = adjacency,
                dropout_rate = dropout_rate if index != num_layers else 0.0,
                activation   = activation if index != num_layers else final_activation)
            self.a_layers.add_module(f'hidden_a{index}', _hidden_layer_a)
            
            _hidden_layer_ppmi = hidden_dense_layer(
                in_features  = n_in,
                out_features = n_out,
                diffusion    = ppmi,
                dropout_rate = dropout_rate if index != num_layers else 0.0,
                W            = _hidden_layer_a.weight,
                activation   = activation if index != num_layers else final_activation)
            self.ppmi_layers.add_module(f'hidden_ppmi{index}', _hidden_layer_ppmi)
        
        
    def forward(self, input : torch.Tensor) -> (torch.Tensor, torch.Tensor):
        output_a = self.a_layers(input)
        output_ppmi = self.ppmi_layers(input)        
        return (output_a, output_ppmi)
    
    
    