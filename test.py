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

import numpy as np
import matplotlib.pyplot as plt

import getdata
import models
import loss
import utilities



def test_DGCN(dataset='cora', epochs=100, learning_rate=1e-3, batch_size=0,
              hidden_size=32, dropout_rate=0.3, L1_reg=0.00, L2_reg=0.00,
              trace=False):
    r"""
    Test the Dual Graph Convolutional Neural Network.
    """
    
    # import data
    G, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = \
        getdata.load_graph_data(dataset, trace)
    
    # get adjacency and ppmi matrices
    adj = G.normalized_adjacency_matrix()
    if trace:   print("  Adjacency matrix: ", tuple(adj.shape))
    
    print("Sampling graph...")
    freq = G.frequency_matrix(path_length=3, num_walks=5, window_size=2)
    if trace:   print("  Frequency matrix: ", tuple(freq.shape))

    ppmi = G.normalized_ppmi_matrix(freq)
    if trace:   print("  PPMI matrix: ", tuple(ppmi.shape))
    
    # setup hidden layer sizes
    feature_size = features.shape[1]
    label_size = y_train.shape[1]
    layer_sizes = [(feature_size, hidden_size), (hidden_size, label_size)]
    if trace:   print("  Hidden layer sizes: ", layer_sizes)
    
    # build graph neural network model
    model = models.DGCN(layer_sizes, adj, ppmi, dropout_rate)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           betas=(0.9, 0.999), eps=1e-8)
    loss_fnc = loss.DualLoss(model.supervised_loss, model.unsupervised_loss)
    
    # TODO: keep track of training loss, validation accurary, validation loss,
    # and testing accurary
    training_losses = []
    validation_losses = []
    validation_accuracies = []
    testing_accuracies = []

    # TODO: implement batch training
    for epoch in range(epochs):
        # forward pass
        y_pred = model(features)
        a_output, ppmi_output = y_pred
        
        # calculate and store loss
        training_loss = loss_fnc(a_output, ppmi_output, y_train, train_mask)
        
        # calculate validation and testing accuracies
        testing_acc = utilities.accuracy(a_output, y_test, test_mask)
        
        # store results for plotting
        training_losses.append(training_loss.item())
        testing_accuracies.append(testing_acc.item())
        
        if trace:
            print("Epoch: {:04n},".format(epoch+1),
                  "train_loss: {:.5f},".format(training_loss.item()),
                  "testing_acc: {:.5f}".format(testing_acc.item())
                  )
            
        # zero gradients, backward pass, update weights
        model.zero_grad()
        training_loss.backward()
        optimizer.step()
    
    return (training_losses, testing_accuracies)
    # return training_losses, validation_losses, validation_accuracies, testing_accuracies

        

def plot_losses(*results, rows, cols):
    # TODO: work in progress
    names = ["training_losses", "supervised_losses", "unsupervised_losses", "testing_accuracies"]
    
    fig = plt.figure(figsize=(10,8))
    for i in range(1, rows*cols + 1):
        xs = np.arange(len(results[i-1]))
        ys = results[i-1]
        fig.add_subplot(rows, cols, i).set_title(names[i-1])
        plt.plot(xs, ys)
    plt.show()
    
    
    
if __name__ == '__main__':
    # dataset = sys.argv[1]
    dataset = 'citeseer'
    print(f"Testing dataset... {dataset}")
    
    if dataset in ['cora', 'citeseer', 'pubmed']:
        result = test_DGCN(dataset=dataset, epochs=100, learning_rate=1e-3, 
                            batch_size=0, hidden_size=32, dropout_rate=0.3, 
                            trace=True)
        plot_losses(*result, rows=1, cols=len(result))
    else:
        print(f"No such a dataset: {dataset}")


