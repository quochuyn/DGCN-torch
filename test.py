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

import getdata
import models
import loss
import utils
import output



def test_DGCN(dataset='cora', data_dir=None, epochs=100, learning_rate=1e-3, 
              batch_size=0, hidden_size=32, dropout_rate=0.3, clip_value=1.0, 
              train_size=0.6, val_size=0.2, test_size=0.2, self_loops=True, 
              path_length=3, num_walks=100, window_size=2, random_state=0, 
              trace=False):
    r"""
    Test the Dual Graph Convolutional Neural Network on a dataset.

    Parameters
    ----------
    dataset : str, optional
        Name of the dataset to be tested on. Several options include 'cora',
        'citeseer', or 'pubmed'. The default is 'cora'.
    data_dir : str, optional
        data_dir : str, optional    
        Directory for the location of the datasets. Default: If None, then use
        the current working directory.
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
    clip_value : float, optional
        Maximum allowed value for the gradients. The gradients are clipped in
        the range [-clip_value, clip_value]. The default is 1.0.
    train_size : float, optional
        Should be between 0.0 and 1.0 and represent the proportion of the
        dataset to include in the train split. The default is 0.6.
    val_size : float, optional
        Should be between 0.0 and 1.0 and represent the proportion of the
        dataset to include in the validation split. The default is 0.2.
    test_size : float, optional
        Should be between 0.0 and 1.0 and represent the proportion of the
        dataset to include in the validation split. The default is 0.2.
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
    random_state : int, optional
        Random state or seed to use. The default is 0.
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
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    np.random.seed(random_state)
    
    # run on gpu if available
    device = utils.get_device()
    
    # import data
    G, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = \
        getdata.load_graph_data(dataset      = dataset, 
                                val_size     = val_size,
                                data_dir     = data_dir,
                                device       = device,
                                random_state = random_state,
                                trace        = trace)
    
    #####################################################
    # layer_sizes = [features.shape[-1], 128, 64]
    # ae = models.Autoencoder(layer_sizes, dropout_rate = dropout_rate,
    #                         final_activation ='tanh').to(device)
    # ae.fit(features, learning_rate = learning_rate, epochs = epochs, 
    #         batch_size = features.shape[0], trace = trace)
    # features = ae(features)
    #####################################################
    
    # get adjacency and ppmi matrices
    adjacency, ppmi = getdata.diffusion_matrices(G           = G, 
                                                 self_loops  = self_loops, 
                                                 path_length = path_length,
                                                 num_walks   = num_walks, 
                                                 window_size = window_size,
                                                 device      = device,
                                                 trace       = trace)
    
    # setup hidden layer sizes
    feature_size = features.shape[-1]
    label_size = y_train.shape[-1]
    layer_sizes = [feature_size, hidden_size, label_size]
    if trace:   print("  Hidden layer sizes: ", layer_sizes)
    
    # build graph neural network model
    model = models.DGCN(layer_sizes  = layer_sizes, 
                        adjacency    = adjacency, 
                        ppmi         = ppmi, 
                        dropout_rate = dropout_rate).to(device)
    optimizer = optim.Rprop(params = model.parameters(), 
                            lr     = learning_rate)
    DualLoss_fnc = loss.DualLoss(supervised_loss_fnc   
                                      = loss.MaskedCrossEntropyLoss(reduction='mean'),
                                 unsupervised_loss_fnc 
                                      = nn.MSELoss(reduction='mean')).to(device)
    
    # keep track of losses and accuracies
    model_results = defaultdict(list)

    for epoch in range(epochs):
        
        for train_batch_mask in utils.yield_batch_mask(train_mask, batch_size, device):
        
            # forward pass
            a_output, ppmi_output = model(features)
            
            # temporal weight function to combine the supervised and
            # unsupervised loss functions within the Dual Loss function
            scaled_unsup_weight_max = utils.get_scaled_unsup_weight_max(
                num_labels=train_mask.sum(), X_train_shape=features.shape[0], 
                unsup_weight_max=15.0)
            ramped_up_weight = utils.rampup(epoch, scaled_unsup_weight_max)
            
            # calculate and store losses and accuracies
            training_loss   = DualLoss_fnc(a_output, ppmi_output, y_train, 
                                           train_batch_mask, ramped_up_weight)
            validation_loss = DualLoss_fnc(a_output, ppmi_output, y_val, 
                                           val_mask, ramped_up_weight)
            training_acc    = utils.accuracy(a_output, y_train, train_mask)
            validation_acc  = utils.accuracy(a_output, y_val, val_mask)
            testing_acc     = utils.accuracy(a_output, y_test, test_mask)
            
            # store results for plotting
            model_results['training_losses'].append(training_loss.item())
            model_results['training_accuracies'].append(training_acc.item())
            model_results['validation_losses'].append(validation_loss.item())
            model_results['validation_accuracies'].append(validation_acc.item())
            model_results['testing_accuracies'].append(testing_acc.item())
        
            if trace:
                print("Epoch: {:04n},".format(epoch+1),
                      "train_loss: {:.5f},".format(training_loss.item()),
                      "val_loss: {:.5f},".format(validation_loss.item()),
                      "train_acc: {:.5f},".format(training_acc.item()),
                      "val_acc: {:.5f},".format(validation_acc.item()),
                      "test_acc: {:.5f}".format(testing_acc.item()))
                
            # zero gradients, backward pass, clip gradients, update weights
            model.zero_grad()
            training_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
    
    return model_results
    
    
    
if __name__ == '__main__':
    # dataset = sys.argv[1]
    dataset = 'ds5'
    print(f"Testing dataset... {dataset}")
    
    if dataset in ['cora', 'citeseer', 'pubmed', 'ds5']:
        model_results = test_DGCN(dataset=dataset, epochs=100, 
                                  learning_rate=0.001, batch_size=0,
                                  hidden_size=32, dropout_rate=0.2, 
                                  clip_value=1.0, train_size=0.6, 
                                  val_size=0.2, test_size=0.2,
                                  self_loops=True, path_length=3,
                                  num_walks=100, window_size=2, trace=True)
        names = ['training_losses', 'validation_losses', 'training_accuracies',
                 'validation_accuracies', 'testing_accuracies']
        output.plot_figures(model_results, names, rows=2, cols=3)
    else:
        print(f"No such a dataset: {dataset}")


