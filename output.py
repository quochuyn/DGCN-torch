# output.py
#
# Author:     Quoc-Huy Nguyen
# Project:    DGCN-torch
# Description:
#             This file contains functions for output-related computations.
# 



import numpy as np
import matplotlib.pyplot as plt



def plot_figures(model_results, names, rows, cols):
    """
    

    Parameters
    ----------
    model_results : dict
        Dictionary of model results.
            'training_losses' : list containing training losses
            'validation_losses' : list containing validation losses
            'validation_accuracies' : list containing validation accuracies
            'testing_accuracies' : list containing testing accuracies
    names : [str]
        List of strings consisting of the keys in `model_results` to plot.
    rows : int
        Number of rows for the figure.
    cols : int
        Number of columns for the figure.
    """
    
    fig = plt.figure(figsize=(12,8))
    for i in range(rows*cols):
        
        # no more items to plot
        if i == len(names):
            break
        
        xs = np.arange(len(model_results[names[i]]))
        ys = model_results[names[i]]
        fig.add_subplot(rows, cols, i + 1).set_title(names[i])
        plt.plot(xs, ys)
    plt.show()


