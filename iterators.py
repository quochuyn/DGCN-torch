# iters.py
#
# Author:     Quoc-Huy Nguyen
# Project:    DGCN-torch
# Description:
#             This file contains iterator functions similar to itertools.
#



def window_pairs(indexable, w):
    '''
    Makes an iterator that yields pairs of each element with neighboring 
    elements that are within w indices away. Does not pair with oneself.
    
    window_pairs('ABCDE', 1) --> AB BA BC CB CD DC DE ED
    window_pairs([1,2,3,4,5], 2) --> 12 13 21 23 24 31 32 34 35 42 43 45 53 54
    '''
    size = len(indexable)
    for i in range(size):
        for j in range(i - w, i + w + 1):
            if i != j and j >= 0 and j < size:
                yield (indexable[i], indexable[j])


def right_window_pairs(indexable, w):
    '''
    Makes an iterator that yields pairs of each element with neighboring
    elements that are within w indices forward. Does not pair with oneself.
    
    forward_window_pairs('ABCDE', 1) --> AB BC CD DE
    forward_window_pairs([1,2,3,4,5], 2) --> 12 13 23 24 34 35 45
    '''
    size = len(indexable)
    for i in range(size):
        for j in range(i, i + w + 1):
            if i != j and j < size:
                yield (indexable[i], indexable[j])
                

                
                