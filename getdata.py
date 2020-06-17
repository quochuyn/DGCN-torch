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
import graphs
import utilities



class FileExtensionError(OSError):
    pass



def _read_binary_file(file : str):
    r"""
    Read a binary file with extension 'x', 'y', 'tx', 'tx', 'allx', 'ally',
    or 'graph'.
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
    Read a file with extension 'index' returning a list of integers specifying 
    indices.
    """
    
    extension = file.split('.')[-1]
    
    if extension == 'index':
        with open(file, 'r') as open_file:
            return [int(line.strip()) for line in open_file]
    else:
        raise FileExtensionError(f"_parse_binary_file: Not sure how to handle file: {file}")



def _fix_citeseer(tx, ty, test_idx_reorder, test_idx_range):
    #TODO
    raise Exception("_fix_citeseer: TODO; Function not implemented yet.")



def load_graph_data(DATASET : str, trace : bool = False):
    r"""
    Load graph data for testing.

    Parameters
    ----------
    DATASET : str
        Name of dataset.
    trace : bool
        Boolean value whether to trace the output.

    Returns
    -------
    G : graphs.graph
        A derived class from networkx.Graph for undirected graphs that includes 
        additional methods for computing graph-related torch.tensor matrices.
    features : torch.tensor
        Feature matrix that includes labeled and unlabeled data.
    y_train : torch.tensor
        Training labels.
    y_val : torch.tensor
        Validation labels.
    y_test : torch.tensor
        Testing labels.
    train_mask : torch.tensor
        Training mask.
    val_mask : torch.tensor
        Validation mask.
    test_mask : torch.tensor
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
    for i in range(len(NAMES)):
        with open('data/ind.{}.{}'.format(DATASET, NAMES[i]), 'rb') as open_file:
            if sys.version_info > (3, 0):
                OBJECTS.append(pickle.load(open_file, encoding='latin1'))
            else:
                OBJECTS.append(pickle.load(open_file))
    x, y, tx, ty, allx, ally, graph = OBJECTS
    
    # test.index : indices of testing instances in graph 
    test_idx_reorder = _read_index_file(f'data/ind.{DATASET}.test.index')
    test_idx_range = sorted(test_idx_reorder)
    
    if DATASET == "citeseer":
        tx, ty = _fix_citeseer(tx, ty, test_idx_reorder, test_idx_range)
    
    # get the Graph: G
    G = graphs.graph(graph)
    
    # get the features: x
    features = torch.cat((torch.as_tensor(allx.toarray()), torch.as_tensor(tx.toarray())))
    features[test_idx_reorder, :] = features[test_idx_range, :]
    if trace:   print("  Feature matrix: ", tuple(features.shape))
    
    # get the labels: y
    labels = torch.cat((torch.as_tensor(ally), torch.as_tensor(ty)))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    if trace:   print("  Label matrix: ", tuple(labels.shape))
    
    # train, validation, test indices
    idx_train = range(len(y))
    idx_val   = range(len(y), len(y) + 500) # should we specify validation size?
    idx_test  = test_idx_range # i dont think tolist() is a torch method
    
    # train, validation, test masks
    train_mask = utilities._sample_mask(idx_train, labels.shape[0])
    val_mask   = utilities._sample_mask(idx_val, labels.shape[0])
    test_mask  = utilities._sample_mask(idx_test, labels.shape[0])
    
    # initialize train, validation, test labels
    y_train = torch.zeros(labels.shape, dtype=torch.int)
    y_val   = torch.zeros(labels.shape, dtype=torch.int)
    y_test  = torch.zeros(labels.shape, dtype=torch.int)
    
    # define train, validation, test labels
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :]   = labels[val_mask, :]
    y_test[test_mask, :]  = labels[test_mask, :]
    
    return G, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
    
    