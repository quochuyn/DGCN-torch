# getdata.py
#
# Author:     Quoc-Huy Nguyen
# Project:    DGCN-torch
# Description:
#             This file contains methods that read and process input data.
#


import sys
import pickle

import torch
import numpy as np
import scipy.sparse as sp

import graphs
import utilities



class FileExtensionError(OSError):
    pass



def _read_binary_file(file : str):
    r"""
    Reads a binary file with extension 'x', 'y', 'tx', 'tx', 'allx', 'ally',
    or 'graph' and returning the pickled file.
    
    Parameters
    ----------
    file : str
        String representation of the file path.
    """
    
    NAMES = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    extension = file.split('.')[-1]
    
    if extension in NAMES:
        with open(file, 'rb') as open_file:
            if sys.version_info > (3, 0):
                 return pickle.load(open_file, encoding='latin1')
            else:
                 return pickle.load(open_file)
    else:
        raise FileExtensionError(f"_parse_binary_file: Not sure how to handle file: {file}")



def _read_index_file(file : str) -> [int]:
    r"""
    Reads a file with extension 'index' and returns a list of integers specifying 
    indices.
    
    Parameters
    ----------
    file : str
        String representation of the file path.
        
    Returns
    -------
    [int]
        List of integers specifying the indices.
    """
    
    extension = file.split('.')[-1]
    
    if extension == 'index':
        with open(file, 'r') as open_file:
            return [int(line.strip()) for line in open_file]
    else:
        raise FileExtensionError(f"_parse_binary_file: Not sure how to handle file: {file}")



def _fix_citeseer(tx, ty, test_idx_reorder, test_idx_range):
    r"""
    Fix citeseet dataset (there are some isolated nodes in the graph)
    Find isolated nodes, add them as zero-vectors into the right position.
    """
    
    test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
    
    tx_extended = sp.lil_matrix((len(test_idx_range_full), tx.shape[1]), dtype=np.float32)
    tx_extended[test_idx_range - min(test_idx_range), :] = tx
    tx = tx_extended
    
    ty_extended = np.zeros((len(test_idx_range_full), ty.shape[1]), dtype=np.int32)
    ty_extended[test_idx_range - min(test_idx_range), :] = ty
    ty = ty_extended
    
    return tx, ty
    


def load_graph_data(DATASET : str, val_size : float, trace : bool = False):
    r"""
    Load graph data for testing.

    Parameters
    ----------
    DATASET : str
        Name of dataset.
    val_size : float
        Should be between 0.0 and 1.0 and represent the proportion of the
        remaining unlabeled training features.
    trace : bool, optional
        Boolean value whether to trace the output. The default value is False.

    Returns
    -------
    G : graphs.graph
        A derived class from networkx.Graph for undirected graphs that includes 
        additional methods for computing graph-related torch.Tensor matrices.
    features : torch.Tensor
        Feature matrix that includes labeled and unlabeled data.
    y_train : torch.Tensor
        Training labels.
    y_val : torch.Tensor
        Validation labels.
    y_test : torch.Tensor
        Testing labels.
    train_mask : torch.Tensor
        Training mask.
    val_mask : torch.Tensor
        Validation mask.
    test_mask : torch.Tensor
        Testing mask.
    """
    
    # x     : training features
    # y     : training labels
    # tx    : testing features
    # ty    : testing labels
    # allx  : all training features (labeled and unlabeled)
    # ally  : all training labels for testing instances in allx
    # graph : dict in the form of {index: [index_of_neighbor_nodes]}
    NAMES = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    OBJECTS = []
    for name in NAMES:
        pickled_file = _read_binary_file(f'data/ind.{DATASET}.{name}')
        OBJECTS.append(pickled_file)
    x, y, tx, ty, allx, ally, graph = OBJECTS

    # test.index : indices of testing instances in graph 
    test_idx_reorder = _read_index_file(f'data/ind.{DATASET}.test.index')
    test_idx_range = np.sort(test_idx_reorder)
    
    if DATASET == "citeseer":
        tx, ty = _fix_citeseer(tx, ty, test_idx_reorder, test_idx_range)
    
    # get the graph: G
    G = graphs.graph(graph)
    
    # get the features: x
    features = torch.cat((torch.as_tensor(allx.toarray()), 
                          torch.as_tensor(tx.toarray())))
    features[test_idx_reorder, :] = features[test_idx_range, :]
    if trace:   print("  Feature matrix: ", tuple(features.shape))
    
    # get the labels: y
    labels = torch.cat((torch.as_tensor(ally), torch.as_tensor(ty)))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    if trace:   print("  Label matrix: ", tuple(labels.shape))
    
    # train, validation, test indices
    train_size = len(y)
    idx_train  = range(train_size)
    idx_val    = range(train_size, train_size + int(val_size * (test_idx_range[0] - train_size)))
    idx_test   = test_idx_range
    
    # train, validation, test masks
    dataset_size = features.shape[0]
    train_mask   = utilities.sample_mask(idx_train, dataset_size)
    val_mask     = utilities.sample_mask(idx_val, dataset_size)
    test_mask    = utilities.sample_mask(idx_test, dataset_size)
    
    # initialize train, validation, test labels
    y_train = torch.zeros(labels.shape, dtype=torch.int)
    y_val   = torch.zeros(labels.shape, dtype=torch.int)
    y_test  = torch.zeros(labels.shape, dtype=torch.int)
    
    # define train, validation, test labels
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :]     = labels[val_mask, :]
    y_test[test_mask, :]   = labels[test_mask, :]
    
    return G, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
    

def diffusion_matrices(G, self_loops : bool, path_length : int, num_walks : int,
                       window_size : int, trace : bool = False):
    r"""
    Given a graph G, returns the two diffusion matrices: the adjacency and ppmi 
    matrices.

    Parameters
    ----------
    G : graphs.graph
        A derived class from networkx.Graph for undirected graphs that includes 
        additional methods for computing graph-related torch.Tensor matrices.
    self_loops : bool
        Condition whether to add self-loops to the adjacency matrix.
    path_length : int
        Length of the random walk used when sampling the graph (i.e.
        computing the frequency matrix).
    num_walks : int
        Number of random walks for each node used when sampling the graph 
        (i.e. computing the frequency matrix).
    window_size : int
        Size of window that subsets the path as it slides along path when
        sampling the graph (i.e. computing the frequency matrix).
    trace : bool, optional
        Boolean value whether to trace the output. The default value is False.

    Returns
    -------
    adjacency : torch.Tensor
        Normalized adjacency matrix of the graph.
    ppmi : torch.Tensor
        Normalized positive pointwise mutual information matrix.
    """
    
    adjacency = G.normalized_adjacency_matrix(self_loops)
    if trace:   print("  Adjacency matrix: ", tuple(adjacency.shape))
    
    print("Sampling graph...")
    ppmi = G.normalized_ppmi_matrix(path_length, num_walks, window_size)
    if trace:   print("  PPMI matrix: ", tuple(ppmi.shape))
    
    return adjacency, ppmi

