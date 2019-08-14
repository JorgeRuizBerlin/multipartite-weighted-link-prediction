#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 12:06:24 2018

@author: paulh
"""
import numpy as np


#def si_scores(x, links, y, *args):
#    """Computes the self information score for the links in x, using
#    the information from x and y."""
#    f_p = args[0]
#    f_mi = args[1]
#    p = f_p(x, links)
#    mi = f_mi(x, links, y, p, *args[2:])
##    assert p.size == mi.size
#    si_scores = mi + np.log2(p)
#    
#    return si_scores


def si_scores(x, links, y, f_p, f_mi, verbose=True):
    """Computes the self information score for each of the 
    given links, using an arbitrary function for the 'a priori'
    probability and the mutual information.
    
    Args:
        x: Incidence matrix of first layer (numpy array)
        links: numpy array with shape (# of links, 2), 
            listing the missing links to condsider.
        y: Incidence matrix of second layer (numpy array)
        f_p: Function to calculate the 'a priori' probability
            of a link existing. Should take only x as its argument.
        f_mi: Function to calculate the mutual information between 
            some topological feature and the existence of each link. 
            Should take as an argument x, links, and the 'a priori'
            probabilities p as a 1D numpy array.
        verbose: Controls printed output. 
        
    Returns:
        si: 1D numpy array, self information scores for each link.
    """
    si = np.zeros(links.shape[0])
    
    p = f_p(x) # 'a priori' probabilities
    mi = f_mi(x, links, y, p, verbose=verbose) # mutual information
    
    si = np.log2(p[links[:,0], links[:,1]]) + mi
            
    return si