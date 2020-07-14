# graphs.py
#
# Author:     Quoc-Huy Nguyen
# Project:    DGCN-torch
# Description:
#             This file contains methods for graph-related computations.
# 


# TODO: 
#   What if nodes are strings not integers?
#   Try to vectorize some of the computations, if there is extra time.


# module contains graph classes
import networkx as nx

import torch
import random
import iterators



class graph(nx.Graph):
    r"""
    Derived class of networkx.Graph for undirected graphs that includes
    aditional methods for computing graph-related torch.Tensor matrices.
    
    The method adjacency_matrix() is overloaded in this derived class to
    return a torch.Tensor matrix instead of a scipy.sparse matrix.
    """
    
    
    def normalized_diffusion_matrix(self, diffusion : torch.Tensor) -> torch.Tensor:
        r"""
        Compute the normalized diffusion matrix, i.e. the adjacency matrix or 
        the positive pointwise mutual information matrix. 
        
        The computation can be described as 
        
            D^{-1/2} A D^{-1/2} 
        
        where `A` is the diffusion matrix and the matrix `D` has diagonal entries 
        D_{i, i} = \sum_{j} A_{i, j}.
        
        Parameters
        ----------
        diffusion : torch.Tensor
            Diffusion matrix of the graph.

        Returns
        -------
        torch.Tensor
            Normalized diffusion matrix.
        """
        
        D_inv_sqrt = diffusion.sum(axis=1).pow(-1/2).diag()
        return D_inv_sqrt.mm(diffusion).mm(D_inv_sqrt)
    
        
    def adjacency_matrix(self, self_loops : bool = False) -> torch.Tensor:
        r"""
        Compute the (binary) adjacency matrix of the graph.
        
        The computation can be described as
            
            A_{i,j} = 1 if self_loops = True or vertices v_i, v_j share an edge
            A_{i,j} = 0 otherwise
        
        Parameters
        ----------
        self_loops : bool
            Condition whether to add self-loops to the adjacency matrix. The
            default value is False.

        Returns
        -------
        adjacency : torch.Tensor
            Adjacency matrix of the graph.
        """
        
        num_nodes = self.number_of_nodes()
        
        # initialize adjacency matrix
        if self_loops == True:
            adjacency = torch.eye(num_nodes, dtype=torch.float32)
        else:
            adjacency = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)

        # for any edge between two nodes, set their position in A as 1
        for node, neighbors in self.adjacency():
            for neighbor in neighbors.keys():
                adjacency[node, neighbor] = 1.0
        
        return adjacency
    
    
    def normalized_adjacency_matrix(self, self_loops : bool = False) -> torch.Tensor:
        r"""
        Compute the normalized (binary) adjacency matrix A of the graph.
        
        The computation can be described as
        
            \tilde{A} = D^{-1/2} A D^{-1/2}
        
        where `A` is the adjacency matrix and `D` is the diagonal degree matrix.
        
        Parameters
        ----------
        self_loops : bool
            Condition whether to add self-loops to the adjacency matrix. The
            default value is False.

        Returns
        -------
        torch.Tensor
            Normalized adjacency matrix of the graph.
        """
        
        adjacency = self.adjacency_matrix(self_loops)
        return self.normalized_diffusion_matrix(adjacency)
        
    
    def _random_walk(self, node : int, path_length : int) -> [int]:
        r"""
        Conduct a random walk over the graph.

        Parameters
        ----------
        node : int
            Starting node of the random walk.
        path_length : int
            Length of the random walk.

        Returns
        -------
        path : [int]
            A list of integers for a random walk path.
        """

        path = [node]
        
        # walk to a random neighbor
        for i in range(path_length - 1):
            neighbors = list(self.neighbors(node))
            
            # if a node has no neighbors, i.e. an isolated node
            if len(neighbors) < 1:
                break
            
            node = random.choice(neighbors)
            path.append(node)
        
        return path
        
    
    def frequency_matrix(self, path_length : int, num_walks : int,
                         window_size : int) -> torch.Tensor:
        r"""
        Compute the frequency matrix of the graph.

        Parameters
        ----------
        path_length : int
            Length of the random walk.
        num_walks : int
            Number of random walks for each node.
        window_size : int
            Size of window that subsets the path as it slides along path.
            
        Returns
        -------
        frequency : torch.Tensor
            Frequency matrix of the graph.
        """
        
        # initialize frequency matrix with zeros
        num_nodes = self.number_of_nodes()
        frequency = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
        
        for node in self.nodes:
            for i in range(num_walks):
                path = self._random_walk(node, path_length)

                # uniformly sample all pairs from path within the window size
                for pair in iterators.right_window_pairs(path, window_size):
                    frequency[pair[0], pair[1]] += 1.0
                    frequency[pair[1], pair[0]] += 1.0

        return frequency

    
    def ppmi_matrix(self, path_length : int, num_walks : int, 
                    window_size : int) -> torch.Tensor:
        r"""
        Compute the positive pointwise mutual information (PPMI) matrix of the 
        graph.
        
        Parameters
        ----------
        path_length : int
            Length of the random walk used when sampling the graph (i.e.
            computing the frequency matrix).
        num_walks : int
            Number of random walks for each node used when sampling the graph 
            (i.e. computing the frequency matrix).
        window_size : int
            Size of window that subsets the path as it slides along path when
            sampling the graph (i.e. computing the frequency matrix).
            
        Returns
        -------
        ppmi : torch.Tensor
            Positive pointwise mutual information matrix.
        """
        
        # compute the frequency matrix (aka sampling the graph)
        frequency = self.frequency_matrix(path_length = path_length, 
                                          num_walks   = num_walks, 
                                          window_size = window_size)
        
        num_rows, num_cols = frequency.size()
        sum_total = frequency.sum()
        
        # estimated probability of each node occuring in each context
        probs = frequency / sum_total
        
        # estimated probability of each node
        row_probs = frequency.sum(axis=1) / sum_total
        
        # estimated probability of each context
        col_probs = frequency.sum(axis=0) / sum_total
        
        z = torch.zeros(num_rows, num_cols)
        pmi = torch.log(probs / (col_probs * row_probs.reshape(-1,1)))
        ppmi = torch.max(pmi, z)

        return ppmi


    def normalized_ppmi_matrix(self, path_length : int, num_walks : int, 
                               window_size : int) -> torch.Tensor:
        r"""
        Compute the normalized positive pointwise mutual information (PPMI)
        matrix of the graph.
        
        The computation can be described as
        
            \tilde^{P} = D^{-1/2} P D^{-1/2}
        
        where `P` is the PPMI matrix and the matrix `D` has diagonal entries 
        D_{i, i} = \sum_{j} A_{i, j}.

        Parameters
        ----------
        path_length : int
            Length of the random walk used when sampling the graph (i.e.
            computing the frequency matrix).
        num_walks : int
            Number of random walks for each node used when sampling the graph 
            (i.e. computing the frequency matrix).
        window_size : int
            Size of window that subsets the path as it slides along path when
            sampling the graph (i.e. computing the frequency matrix).

        Returns
        -------
        torch.Tensor
            Normalized positive pointwise mutual information matrix.
        """
        
        ppmi = self.ppmi_matrix(path_length = path_length, 
                                num_walks   = num_walks, 
                                window_size = window_size)
        return self.normalized_diffusion_matrix(ppmi)


