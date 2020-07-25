# getdata.py
#
# Author:     Quoc-Huy Nguyen
# Project:    DGCN-torch
# Description:
#             This file contains methods that read and process input data.
#


import sys
import pickle
from pathlib import Path

import torch
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from math import ceil

import graphs
import utils



class FileExtensionError(OSError):
    pass



def load_graph_data(dataset : str, data_dir : str = None, 
                    train_size : float = 0.6, val_size : float = 0.2, 
                    test_size : float = 0.2, device : torch.device = None, 
                    random_state : int = 0, trace : bool = False):
    r"""
    Load graph data for testing.

    Parameters
    ----------
    dataset : str
        Name of dataset.
    data_dir : str, optional    
        Directory for the location of the datasets. Default: If None, then use
        the current working directory.
    val_size : float, optional
        Should be between 0.0 and 1.0 and represent the proportion of the
        remaining unlabeled training features. The default is 0.5.
    device : torch.device, optional
        The device to run the computations on (i.e. the gpu or cpu). 
        Default: If None, then uses the current device for the default
        tensor type (see torch.set_default_tensor_type()).
    random_state : int, optional
        Random state or seed to use. The default is 0.
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
    
    if dataset in ['cora', 'citeseer', 'pubmed']:
        return _load_graph_data(dataset      = dataset,
                                data_dir     = data_dir,
                                device       = device,
                                random_state = random_state,
                                trace        = trace)
    elif dataset == 'ds5':
        return _load_sc_graph_data(dataset      = dataset,
                                   train_size   = train_size,
                                   val_size     = val_size,
                                   test_size    = test_size,
                                   data_dir     = data_dir,
                                   device       = device,
                                   random_state = random_state,
                                   trace        = trace)
    


def _load_graph_data(dataset : str, data_dir : str = None, 
                     device : torch.device = None, random_state : int = 0, 
                     trace : bool = False):
    r"""
    Load graph data for testing. Supported datasets include 'cora', 'citeseer',
    and 'pubmed'.
    """
    
    # x     : training features
    # y     : training labels
    # tx    : testing features
    # ty    : testing labels
    # allx  : all training features (labeled and unlabeled)
    # ally  : all training labels for testing instances in allx
    # graph : dict in the form of {index: [index_of_neighbor_nodes]}
    # test.index : indices of testing instances in graph
    x, y, tx, ty, allx, ally, graph, test_idx_reorder, test_idx_range = \
        _read_binary_files(dataset, data_dir)
    
    # get the graph: G
    G = graphs.graph(graph)
    
    # get the features: x
    features = torch.cat((torch.as_tensor(allx.toarray(), device=device),
                          torch.as_tensor(tx.toarray(), device=device)))
    features[test_idx_reorder, :] = features[test_idx_range, :]
    if trace:   print("  Feature matrix: ", tuple(features.shape))
    
    # get the labels: y
    labels = torch.cat((torch.as_tensor(ally, device=device), 
                        torch.as_tensor(ty, device=device)))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    if trace:   print("  Label matrix: ", tuple(labels.shape))
    
    # get sizes
    train_size   = len(y)
    test_size    = len(ty)
    dataset_size = features.shape[0]
    
    # train, validation, test indices
    idx_train  = range(train_size)
    idx_val    = range(train_size, dataset_size - test_size)
    idx_test   = test_idx_range
    
    y_train, y_val, y_test, train_mask, val_mask, test_mask = \
        _get_labels_and_masks(labels, idx_train, idx_val, idx_test, device)
    
    return G, features, y_train, y_val, y_test, train_mask, val_mask, test_mask



def _load_sc_graph_data(dataset : str, data_dir : str = None, 
                        train_size : float = 0.6, val_size : float = 0.2, 
                        test_size : float = 0.2, device : torch.device = None, 
                        random_state : int = 0, trace : bool = False):
    r"""
    Load single-cell RNA-sequencing graph data for testing. Supported datasets
    include 'ds5'.
    """
    
    assert type(train_size) is float, \
        f"_load_sc_graph_data: expected float type for train_size argument, but got {type(train_size)}"
    assert type(val_size) is float, \
        f"_load_sc_graph_data: expected float type for val_size argument, but got {type(val_size)}"
    assert type(test_size) is float, \
        f"_load_sc_graph_data: expected float type for test_size argument, but got {type(test_size)}"
    assert train_size + val_size + test_size == 1.0, \
        f"_load_sc_graph_data: invalid combination of train_size ({train_size}), val_size ({val_size}), and test_size ({test_size})"
    
    G, features, labels = _load_mat_files(dataset, data_dir, device)
    
    # shuffle data into training and testing examples
    X_train, X_test, Y_train, Y_test = train_test_split(features, labels,
                                                        train_size   = train_size + val_size,
                                                        test_size    = test_size,
                                                        random_state = random_state,
                                                        shuffle      = True,
                                                        stratify     = labels)
    
    features = torch.cat((X_train, X_test))
    if trace:   print("  Feature matrix: ", tuple(features.shape))
    labels = torch.cat((Y_train, Y_test)).int()
    if trace:   print("  Label matrix: ", tuple(labels.shape))
    
    # get sizes
    train_val_sz = X_train.shape[0]
    train_sz = int((train_size / (train_size + val_size)) * train_val_sz)
    val_sz   = ceil((val_size / (train_size + val_size)) * train_val_sz)
    test_sz  = X_test.shape[0]
    dataset_sz = labels.shape[0]
    
    # make sure we dont leave out some examples
    assert train_sz + val_sz == train_val_sz, f"train_sz ({train_sz}), val_sz ({val_sz}), train_val_sz ({train_val_sz})"
    assert dataset_sz - test_sz == val_sz + train_sz
    
    # train, validation, test indices
    idx_train  = range(train_sz)
    idx_val    = range(train_sz, val_sz + train_sz)
    idx_test   = range(val_sz + train_sz, dataset_sz)
    
    y_train, y_val, y_test, train_mask, val_mask, test_mask = \
        _get_labels_and_masks(labels, idx_train, idx_val, idx_test, device)
    
    return G, features, y_train, y_val, y_test, train_mask, val_mask, test_mask



def diffusion_matrices(G, self_loops, path_length, num_walks, window_size, 
                       device = None, trace = False):
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
    device : torch.device, optional
        The device to run the computations on (i.e. the gpu or cpu). 
        Default: If None, then uses the current device for the default
        tensor type (see torch.set_default_tensor_type()).
    trace : bool, optional
        Boolean value whether to trace the output. The default value is False.

    Returns
    -------
    adjacency : torch.Tensor
        Normalized adjacency matrix of the graph.
    ppmi : torch.Tensor
        Normalized positive pointwise mutual information matrix.
    """
    
    adjacency = G.normalized_adjacency_matrix(self_loops, device, trace)
    if trace:   print("  Adjacency matrix: ", tuple(adjacency.shape))
    
    if trace:   print("Sampling graph...")
    ppmi = G.normalized_ppmi_matrix(path_length, num_walks, window_size, 
                                    device, trace)
    if trace:   print("  PPMI matrix: ", tuple(ppmi.shape))
    
    return adjacency, ppmi



def _read_binary_files(dataset : str, data_dir : str = None):
    r"""
    Reads the binary files with extension 'x', 'y', 'tx', 'ty', 'allx', 'ally',
    and 'graph' corresponding to the dataset in the data_dir(ectory).

    Parameters
    ----------
    dataset : str
        Name of dataset.
    data_dir : str, optional    
        Directory for the location of the datasets. Default: If None, then use
        the current working directory.

    Returns
    -------
    x : scipy.sparse.csr.csr_matrix
        Training features.
    y : numpy.ndarray
        Training labels.
    tx : scipy.sparse.csr.csr_matrix
        Testing features.
    ty : numpy.ndarray
        Testing labels.
    allx : scipy.sparse.csr.csr_matrix
        All training features (labeled and unlabeled).
    ally : numpy.ndarray
        All training labels for testing instances in allx.
    graph : defaultdict
        Dictionary in the form of {index : [index_of_neighbor_nodes]}
    test_idx_reorder : list
        Unsorted test indices.
    test_idx_range : list
        Sorted test indices.
    """
    
    NAMES = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    OBJECTS = []
    directory = '' if data_dir is None else data_dir + '/'
    for name in NAMES:
        file = Path(directory + f'data/{dataset}/ind.{dataset}.{name}')
        pickled_file = _read_binary_file(file)
        OBJECTS.append(pickled_file)
    x, y, tx, ty, allx, ally, graph = OBJECTS
    
    # test.index : indices of testing instances in graph
    index_file = Path(directory + f'data/{dataset}/ind.{dataset}.test.index')
    test_idx_reorder = _read_index_file(index_file)
    test_idx_range = sorted(test_idx_reorder)
    
    if dataset == 'citeseer':
        tx, ty = _fix_citeseer(tx, ty, test_idx_reorder, test_idx_range)
    
    return x, y, tx, ty, allx, ally, graph, test_idx_reorder, test_idx_range



def _load_mat_files(dataset : str, data_dir : str = None, 
                    device : torch.device = None):
    r"""
    Load MATLAB .mat files and get the graph G, features and labels.
    """
    
    # read MATLAB .mat files
    directory = '' if data_dir is None else data_dir + '/'
    graph = loadmat(Path(directory + f'data/{dataset}/graph_{dataset}.mat'))
    data = loadmat(Path( directory + f'data/{dataset}/data_{dataset}.mat'))
    
    G = graphs.graph(graph['sort_W'])
    features = torch.as_tensor(data['counts'].toarray(), dtype=torch.float32,
                               device=device)
    labels = torch.as_tensor(data['cluster_label'], dtype=torch.long,
                             device=device)

    # fix labels to be appropriate shape, then fill in 1's at the column
    # corresponding to the unique label; labels should start at 0, not 1
    dataset_size = labels.shape[0]
    new_labels = torch.zeros((dataset_size, torch.unique(labels).shape[0]))
    row_range = torch.arange(dataset_size).reshape(-1,1)
    indices = torch.cat((row_range, labels-1), dim=1)
    new_labels[indices[:,0], indices[:,1]] = 1
    labels = new_labels
    
    return G, features, labels



def _read_binary_file(file):
    r"""
    Reads a binary file with extension 'x', 'y', 'tx', 'ty', 'allx', 'ally',
    or 'graph' and returning the pickled file.
    """
    
    EXTENSIONS = ['.x', '.y', '.tx', '.ty', '.allx', '.ally', '.graph']
    extension = file.suffix
    
    if extension in EXTENSIONS:
        with open(file, 'rb') as open_file:
            if sys.version_info > (3, 0):
                 return pickle.load(open_file, encoding='latin1')
            else:
                 return pickle.load(open_file)
    else:
        raise FileExtensionError(f"_parse_binary_file: Not sure how to handle file: {file}")



def _read_index_file(file):
    r"""
    Reads a file with extension 'index' and returns a list of integers 
    specifying indices.
    """
    
    extension = file.suffix
    
    if extension == '.index':
        with open(file, 'r') as open_file:
            return [int(line.strip()) for line in open_file]
    else:
        raise FileExtensionError(f"_parse_binary_file: Not sure how to handle file: {file}")



def _get_labels_and_masks(labels, idx_train, idx_val, idx_test, 
                          device : torch.device = None):
    r"""
    Get the labels and masks for the training, validation, and testing sets.
    """
    
    dataset_size = labels.shape[0]
    
    # train, validation, test masks
    train_mask   = utils.sample_mask(idx_train, dataset_size, device=device)
    val_mask     = utils.sample_mask(idx_val, dataset_size, device=device)
    test_mask    = utils.sample_mask(idx_test, dataset_size, device=device)
    
    # initialize train, validation, test labels
    y_train = torch.zeros(labels.shape, dtype=torch.int, device=device)
    y_val   = torch.zeros(labels.shape, dtype=torch.int, device=device)
    y_test  = torch.zeros(labels.shape, dtype=torch.int, device=device)
    
    # define train, validation, test labels
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :]     = labels[val_mask, :]
    y_test[test_mask, :]   = labels[test_mask, :]
    
    return y_train, y_val, y_test, train_mask, val_mask, test_mask



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
    

    
if __name__ == '__main__':
    
    
    
    G, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = \
        _load_sc_graph_data(dataset      = 'ds5',
                            data_dir     = None,
                            train_size   = 0.8,
                            val_size     = 0.0,
                            test_size    = 0.2,
                            device       = None,
                            random_state = 0,
                            trace        = True)
    
    def get_class_counts(tensor):
        return tensor.sum(dim=0)
    
    def get_class_proportions(tensor):
        class_counts = get_class_counts(tensor).float()
        return class_counts / tensor.sum()
    
    print(get_class_counts(y_train))
    print(get_class_counts(y_val))
    print(get_class_counts(y_test))
        
    print(get_class_proportions(y_train))
    print(get_class_proportions(y_val))
    print(get_class_proportions(y_test))
    
    