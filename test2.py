#  test2.py

import sys
import time
import pickle

import scipy.sparse as sp
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

import matplotlib.pyplot as plt

import graphs
import models
import getdata
import layers



def type_as_str(obj):
    # return the type of the object as a string
    return str(type(obj))[8:-2]


def print_figures(matrices, names, rows, cols):
    fig = plt.figure(figsize=(10,10))
    for i in range(1, rows*cols + 1):
        fig.add_subplot(rows, cols, i).set_title(names[i-1])
        plt.imshow(matrices[i-1], cmap='hot')
    plt.show()
    

def from_sparse_to_tensor(sparse):
    # Convert scipy sparse matrix to torch.tensor
    return torch.as_tensor(sparse.toarray())


def load_karate_network():
    G = graphs.graph(nx.karate_club_graph())
    return G


def main():
    
    # G, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = getdata.load_graph_data("cora")
    
    G = load_karate_network()
    
    # adjacency, frequency, positive pointwise mutual information matrices
    A = G.normalized_adjacency_matrix(self_loops=True)
    F = G.frequency_matrix(path_length=3, num_walks=100, window_size=2)
    P = G.normalized_ppmi_matrix(F)
    
    # print_figures([A,F,P], ['Adjacency', 'Frequency', 'PPMI'], rows=1, cols=3)
    # print_figures([A,P], ['Adjacency', 'PPMI'], rows=1, cols=2)
    nx.draw(G)



def main2():
    # mess around with the actual input data
    DATASET = "cora"
    NAMES = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    OBJECTS = []
    for i in range(len(NAMES)):
        with open('data/ind.{}.{}'.format(DATASET, NAMES[i]), 'rb') as open_file:
            if sys.version_info > (3, 0):
                OBJECTS.append(pickle.load(open_file, encoding='latin1'))
            else:
                OBJECTS.append(pickle.load(open_file))
    x, y, tx, ty, allx, ally, graph = tuple(OBJECTS)
    # x     (140, 1433)  -> x_train (labeled)
    # y     (140, 7)     -> y_train
    # tx    (1000, 1433) -> x_test
    # ty    (1000, 7)    -> y_test
    # allx  (1708, 1433) -> data (labeled + unlabeled)
    # ally  (1708, 7)    -> labels (unlabeled cells just have 0)
    # graph : {index : [index_of_neighbor_nodes]}


def main3():
    # mess around with the result of load_graph_data
    DATASET = "cora"
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = dp.load_graph_data(DATASET)
    
    NAMES = ['adj', 'features', 'y_train', 'y_val', 'y_test', 'train_mask', 'val_mask', 'test_mask']    
    # for i in range(len(NAMES)):
        # print(NAMES[i], str(type(eval(NAMES[i])))[8:-2])
        # adj       (2708, 2708) scipy.sparse.csr.csr_matrix
        # features  (2708, 1433) scipy.sparse.lil.lil_matrix
        # y_train   (2708, 7)    numpy.ndarray
        # y_val     (2708, 7)    numpy.ndarray
        # y_test    (2708, 7)    numpy.ndarray
        # train_mask(2708,)      numpy.ndarray
        # val_mask  (2708,)      numpy.ndarray
        # test_mask (2708,)      numpy.ndarray

def main4():

    import pstats
    import cProfile


    cProfile.run('runtask()', 'profile')
    p = pstats.Stats('profile')
    
    p.strip_dirs().sort_stats('time').print_stats(10)


def main5():
    dataset = 'cora'
    trace = True
    G, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = \
        getdata.load_graph_data(dataset, trace)
        
    pos = nx.spring_layout(G, iterations=200)
    nx.draw(G, pos, node_color=range(2708), node_size=2708, cmap=plt.cm.Blues)
    plt.show()
    


if __name__ == '__main__':
    print("Begin Testing...")
    
    print(int(5.5))
    
    print("End Testing...")
    