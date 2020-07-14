# test.py
#
# Author:     Quoc-Huy Nguyen
# Project:    DGCN-torch
# Description:
#             This file contains a script for testing DGCN on a dataset.
# 



import sys
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import getdata
import models
import loss
import utilities



def test_DGCN(dataset='cora', epochs=100, learning_rate=1e-3, batch_size=0,
              hidden_size=32, dropout_rate=0.3, val_size=0.5, self_loops=True, 
              path_length=3, num_walks=100, window_size=2, trace=False):
    r"""
    Test the Dual Graph Convolutional Neural Network on a dataset.

    Parameters
    ----------
    dataset : str, optional
        Name of the dataset to be tested on. Several options include 'cora',
        'citeseer', or 'pubmed'. The default is 'cora'.
    epochs : int, optional
        Number of epochs or iterations to train the model on. The default is 
        100.
    learning_rate : float, optional
        Learning rate that is passed into the optimizer. The default is 1e-3.
    batch_size : int, optional
        The size of a single training batch. If the value is 0, then full batch
        training is assumed. The default is 0.
    hidden_size : int, optional
        The size of the hidden layer. The default is 32.
    dropout_rate : float, optional
        Dropout rate at each layer. The default is 0.3.
    val_size : float, optional
        Should be between 0.0 and 1.0 and represent the proportion of the
        remaining unlabeled training features.
    self_loops : bool, optional
        Condition whether to add self-loops to the adjacency matrix. The
        default is True.
    path_length : int, optional
        Length of the random walk used when sampling the graph (i.e.
        computing the frequency matrix). The default is 3.
    num_walks : int, optional
        Number of random walks for each node used when sampling the graph 
        (i.e. computing the frequency matrix). The default is 100.
    window_size : int, optional
        Size of window that subsets the path as it slides along path when
        sampling the graph (i.e. computing the frequency matrix). The default 
        is 2.
    trace : bool, optional
        Boolean value whether to trace the output. The default is False.

    Returns
    -------
    model_results : dict
        A dictionary containing the training & validation losses and the 
        validation & testing accuracies.
    """
    
    # set random seed for reproducibility although there are a few pytorch
    # operations that are non-deterministic, but none of those are present here
    torch.manual_seed(0)
    
    # import data
    G, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = \
        getdata.load_graph_data(dataset, val_size, trace)
    
    # get adjacency and ppmi matrices
    adjacency, ppmi = getdata.diffusion_matrices(G           = G, 
                                                 self_loops  = self_loops, 
                                                 path_length = path_length,
                                                 num_walks   = num_walks, 
                                                 window_size = window_size,
                                                 trace       = trace)
    
    # setup hidden layer sizes
    feature_size = features.shape[1]
    label_size = y_train.shape[1]
    layer_sizes = [(feature_size, hidden_size), (hidden_size, label_size)]
    if trace:   print("  Hidden layer sizes: ", layer_sizes)
    
    # build graph neural network model
    model = models.DGCN(layer_sizes  = layer_sizes, 
                        adjacency    = adjacency, 
                        ppmi         = ppmi, 
                        dropout_rate = dropout_rate)
    optimizer = optim.Adam(params = model.parameters(), 
                           lr     = learning_rate,
                           betas  = (0.9, 0.999), 
                           eps    = 1e-8)
    # loss_fnc = loss.DualLoss(supervised_loss_fnc   = loss.MaskedMSELoss(reduction='mean'),
    #                           unsupervised_loss_fnc = nn.MSELoss(reduction='mean'))
    loss_fnc = loss.DualLoss(supervised_loss_fnc   = loss.MaskedCrossEntropyLoss(reduction='mean'),
                             unsupervised_loss_fnc = nn.MSELoss(reduction='mean'))
    
    # keep track of losses and accuracies
    model_results = defaultdict(list)

    for epoch in range(epochs):
        
        for train_batch_mask in utilities.yield_batch_mask(train_mask, batch_size):
        
            # forward pass
            a_output, ppmi_output = model(features)
            
            # calculate and store losses and accuracies
            training_loss = loss_fnc(a_output, ppmi_output, y_train, train_batch_mask)
            validation_loss = loss_fnc(a_output, ppmi_output, y_val, val_mask)
            validation_acc = utilities.accuracy(a_output, y_val, val_mask)
            testing_acc = utilities.accuracy(a_output, y_test, test_mask)
            
            # store results for plotting
            model_results['training_losses'].append(training_loss.item())
            model_results['validation_losses'].append(validation_loss.item())
            model_results['validation_accuracies'].append(validation_acc.item())
            model_results['testing_accuracies'].append(testing_acc.item())
            
            if trace:
                print("Epoch: {:04n},".format(epoch+1),
                      "train_loss: {:.5f},".format(training_loss.item()),
                      "val_loss: {:.5f},".format(validation_loss.item()),
                      "val_acc: {:.5f},".format(validation_acc.item()),
                      "test_acc: {:.5f}".format(testing_acc.item()))
                
            # zero gradients, backward pass, update weights
            model.zero_grad()
            training_loss.backward()
            optimizer.step()
    
    return model_results
    
        

def plot_figures(model_results, rows, cols):
    NAMES = ['training_losses', 'validation_losses', 'validation_accuracies', 
             'testing_accuracies', ]
    
    fig = plt.figure(figsize=(12,8))
    for i in range(rows*cols):
        xs = np.arange(len(model_results[NAMES[i]]))
        ys = model_results[NAMES[i]]
        fig.add_subplot(rows, cols, i + 1).set_title(NAMES[i])
        plt.scatter(xs, ys, s=0.5)
    plt.show()
    
    
    
if __name__ == '__main__':
    # dataset = sys.argv[1]
    dataset = 'cora'
    print(f"Testing dataset... {dataset}")
    
    if dataset in ['cora', 'citeseer', 'pubmed']:
        model_results = test_DGCN(dataset=dataset, epochs=100, 
                                  learning_rate=1e-3, batch_size=0,
                                  hidden_size=32, dropout_rate=0.0, 
                                  val_size=0.5, self_loops=True, path_length=3,
                                  num_walks=100, window_size=2, trace=True)
        plot_figures(model_results, rows=2, cols=2)
    else:
        print(f"No such a dataset: {dataset}")


