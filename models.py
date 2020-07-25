# models.py
#
# Author:     Quoc-Huy Nguyen
# Project:    DGCN-torch
# Description:
#             This is a PyTorch implementation of DGCN, a Dual Graph 
#             Convolutional Neural Network for graph-based semi-supervised 
#             classification described in the Chenyi Zhuang and Qiang Ma paper.
# 



from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from layers import hidden_dense_layer



class DGCN(nn.Module):
    r"""
    The Dual Graph Convolutional Neural Network (DGCN) is a model for 
    graph-based semi-supervised classifation. It combines two neural
    networks that simultaneously learns the `local consistency-based knowledge' 
    and the `global consistency-based knowledge' of the graph structured data. 
    Respectively, the adjacency and positive pointwise mutual information 
    (PPMI) matrices capture these knowledges.

    Parameters
    ----------
    layer_sizes : list
        List of layer sizes from input size to output size.
    adjacency : torch.Tensor
        Adjacency matrix of the graph.
    ppmi : torch.Tensor
        Positive pointwise mutual information matrix of the graph.
    dropout_rate : float, optional
        Dropout rate at each layer. The default is 0.3.
    activation : str, optional
        The activation function for the hidden layers. The default is 'relu'.
    final_activation : str, optional
        The final activation function for the output layer. The default is 
        'softmax'.
    """
    
    
    def __init__(self, layer_sizes, adjacency, ppmi, dropout_rate=0.3, 
                 activation='relu', final_activation='softmax'):
        super(DGCN, self).__init__()
        self.a_layers = nn.Sequential()
        self.ppmi_layers = nn.Sequential()
        
        # define the dual NN sharing the same weight W
        num_layers = len(layer_sizes) - 1   # excluding input layer
        for l in range(num_layers):
            _hidden_layer_a = hidden_dense_layer(
                in_features  = layer_sizes[l],
                out_features = layer_sizes[l+1],
                diffusion    = adjacency,
                dropout_rate = dropout_rate if l+1 != num_layers else 0.0,
                # batch_norm   = True if l+1 != num_layers else False,
                activation   = activation if l+1 != num_layers else final_activation)
            self.a_layers.add_module(f'hidden_a{l}', _hidden_layer_a)
            
            _hidden_layer_ppmi = hidden_dense_layer(
                in_features  = layer_sizes[l],
                out_features = layer_sizes[l+1],
                diffusion    = ppmi,
                dropout_rate = dropout_rate if l+1 != num_layers else 0.0,
                W            = _hidden_layer_a.weight,
                batch_norm   = True if l+1 != num_layers else False,
                activation   = activation if l+1 != num_layers else final_activation)
            self.ppmi_layers.add_module(f'hidden_ppmi{l}', _hidden_layer_ppmi)
        
        
    def forward(self, input : torch.Tensor) -> (torch.Tensor, torch.Tensor):
        output_a = self.a_layers(input)
        output_ppmi = self.ppmi_layers(input)        
        return (output_a, output_ppmi)
    


class Autoencoder(nn.Module):
    r"""
    Autoencoder.   

    Parameters
    ----------
    layer_sizes : list
        List of layer sizes from input size to the hidden size for the encoder.
        The layer sizes are reversed for the decoder.
    dropout_rate : float, optional
         Dropout rate at each layer. The default is 0.3.
    activation : str, optional
         The activation function for the hidden layers. The default is 'relu'.
    final_activation : str, optional
         The final activation function for the output layer. The default is 
         'tanh'.
    """
    
    def __init__(self, layer_sizes, dropout_rate=0.3, activation='relu', 
                 final_activation='tanh'):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        
        num_stacks = len(layer_sizes) - 1
        num_layers = 2 * num_stacks   # excluding input layer
        for l in range(num_stacks):
            _hidden_layer_e = hidden_dense_layer(
                in_features  = layer_sizes[l],
                out_features = layer_sizes[l+1],
                dropout_rate = dropout_rate,
                batch_norm   = True,
                activation   = activation)
            self.encoder.add_module(f'hidden_encoder{l}', _hidden_layer_e)
        
        for l in range(num_stacks, 0, -1):
            _hidden_layer_d = hidden_dense_layer(
                in_features  = layer_sizes[l],
                out_features = layer_sizes[l-1],
                dropout_rate = dropout_rate if l-1 != 0 else 0.0,
                batch_norm   = True if l-1 != 0 else False,
                activation   = activation if l-1 != 0 else final_activation)
            self.decoder.add_module(f'hidden_decoder{l}', _hidden_layer_d)
            
            
    def forward(self, input : torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(input))
    
    
    def representations(self, input : torch.Tensor) -> torch.Tensor:
        r"""
        Returns the (lower-dimensional) latent space embeddings of the input 
        features.
        """
        return self.encoder(input)
    
    
    def fit(self, input : torch.Tensor, learning_rate : float, epochs : int, 
            batch_size : int, trace : bool = False):
        dataset = torch.utils.data.TensorDataset(input)
        dataloader = torch.utils.data.DataLoader(dataset, 
                                                 batch_size = batch_size)

        # setup model
        # loss_fnc = nn.BCEWithLogitsLoss(reduction='mean')
        loss_fnc = nn.MSELoss(reduction='mean')
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        model_results = defaultdict(list)
        for epoch in range(epochs):
            for data in dataloader:
                x = data[0]
                y = self(x)
                
                loss = loss_fnc(y, x)
                
                model_results['training_losses'].append(loss.item())
                
                if trace:
                    print("Epoch: {:04n}".format(epoch+1),
                          "training_loss: {:.5f}".format(loss.item()))
                
                self.zero_grad()
                loss.backward()
                optimizer.step()
                
        return model_results
            
    
    
if __name__ == '__main__':
    import numpy as np
    import getdata
    import output
    import utils
    
    # set random seed for reproducibility although there are a few pytorch
    # operations that are non-deterministic, but none of those are present here
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    
    # run on gpu if available
    device = utils.get_device()
    dataset = 'cora'
    val_size = 0.5
    trace = True
    
    # import data
    G, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = \
        getdata.load_graph_data(dataset  = dataset, 
                                val_size = val_size,
                                device   = device,
                                trace    = trace)
    # hyperparameters
    dropout_rate = 0.2
    learning_rate = 1e-3
    epochs = 100
    batch_size = features.shape[0]
    layer_sizes = [features.shape[-1], 256, 128]
    
    # autoencoder
    ae = Autoencoder(layer_sizes, dropout_rate = dropout_rate).to(device)
    model_results = ae.fit(features, learning_rate = learning_rate, 
                           epochs = epochs, batch_size = batch_size, 
                           trace = trace)
    
    # print results
    names = ['training_losses']
    output.plot_figures(model_results, names, rows=1, cols=1)
    
    
    