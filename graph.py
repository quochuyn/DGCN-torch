# graph.py
#
# Author:     Quoc-Huy Nguyen
# Project:    DGCN-torch
# Description:
#             This file contains methods for graph-related computations.
# 


# TODO: 
#   What if nodes are str not int?


# module contains graph classes
import networkx as nx

import torch
import random
import iters



class graph(nx.Graph):
        
    def compute_adjacency(self, self_loops : bool = True) -> torch.tensor:
        '''
        Compute the (binary) adjacency matrix A of the graph.
        
        Parameters
        ----------
        self_loops : bool
            Condition whether to add self-loops to the adjacency matrix.

        Returns
        -------
        A : torch.tensor
            Adjacency matrix of the graph.
        '''

        
        num_nodes = self.number_of_nodes()
        
        # initialize adjacency matrix
        if self_loops == True:
            A = torch.eye(num_nodes, dtype=torch.float32)
        else:
            A = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)

        # for any edge between two nodes, set their position in A as 1
        for node, neighbors in self.adjacency():
            for neighbor in neighbors.keys():
                A[node, neighbor] = 1.0
        
        return A
    
    
    def compute_normalized_adjacency(self, self_loops : bool = True) -> torch.tensor:
        '''
        Compute the normalized adjacency matrix A of the graph.

        Parameters
        ----------
        self_loops : bool
            Condition whether to add self-loops to the adjacency matrix.
        '''
        
        A = self.compute_adjacency(self_loops)
        
        # degree matrix ^ (-1/2)
        D = A.sum(axis = 1).pow(-1/2).diag() 
        
        return D.mm(A).mm(D)
    
    
    def compute_frequency(self, adjacency : torch.tensor, path_length : int,
                          num_walks : int, window_size : int) -> torch.tensor:
        '''
        Compute the frequency matrix F of the graph. In the paper, the
        window size w is used to determine how to sample the node pairs from
        the path, but a window size other than 2 does not make sense.

        Parameters
        ----------
        adjacency : torch.tensor
            Adjacency matrix of the graph.
        path_length : int
            Length of the random walk.
        num_walks : int
            Number of random walks for each node.
        window_size : int
            Size of window that subsets the path as it slides along.
            
        Returns
        -------
        F : torch.tensor
            Frequency matrix of the graph.
        '''
        
        # initialize frequency matrix with zeros
        num_nodes = self.number_of_nodes()
        F = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
        
        for node in self.nodes:
            for i in range(num_walks):
                path = self.random_walk(node, path_length)

                # uniformly sample all pairs from path within the window size
                for pair in iters.window_pairs(path, window_size):
                    if pair[0] == pair[1]:
                        F[pair[0], pair[1]] += 1.0
                    else:
                        F[pair[0], pair[1]] += 1.0
                        F[pair[1], pair[0]] += 1.0

        return F
    
    
    def random_walk(self, node : int, path_length : int) -> list:
        '''
        Conduct a random walk over the graph.

        Parameters
        ----------
        node : int
            Starting node of the random walk.
        path_length : int
            Length of the random walk.

        Returns
        -------
        path : list
            Random walk path.
        '''

        path = [node]
        
        # walk to a random neighbor
        for i in range(path_length - 1):
            neighbors = list(self.neighbors(node))
            
            # assume undirected graph, so not necessary
            # if len(neighbors) < 1:
            #     break
            neighbor = random.choice(neighbors)
        
            node = neighbor
            path.append(node)
        
        return path
            
    
    def compute_PPMI(self, frequency : torch.tensor) -> torch.tensor:
        '''
        Compute the positive pointwise mutual information PPMI matrix of the 
        graph. 

        Parameters
        ----------
        frequency : torch.tensor
            Frequency matrix of the graph.
        '''
        
        num_rows, num_cols = frequency.size()
        sum_total = frequency.sum()
        
        # estimated probability of each node occuring in each context
        prob = frequency / sum_total
        
        # estimated probability of each node
        row_prob = frequency.sum(axis=1) / sum_total
        
        # estimated probability of each context
        col_prob = frequency.sum(axis=0) / sum_total
        
        z = torch.zeros(num_rows, num_cols)
        pmi = torch.log(prob / (col_prob * row_prob.view(-1,1)))
        
        return torch.max(pmi, z)

    
    