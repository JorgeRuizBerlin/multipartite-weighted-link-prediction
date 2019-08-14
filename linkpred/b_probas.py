#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 12:58:18 2018

@author: paulherringer
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

def proba_dense(x):
    """Uses the link density of the network as a global link probability."""
    p = np.sum(np.triu(x, 1)) / (x.shape[0]*(x.shape[0]-1) / 2)
    
    return p


def proba_comb_links(x, links, plot=False):
    """Computes the probability of the given links, given that we 
    know the number of links in the network and the degree pair. We have 
    found counterexamples to show that this probability is not really 
    correct, but it still works fairly well. 
    """
    M = np.sum(np.triu(x, 1)) # number of links
    deg_a = np.sum(x, axis=1)
    deg_b = np.sum(x, axis=0)
    
    ka = deg_a[links[:,0]] # degrees of the first node in each link
    kb = deg_b[links[:,1]] # degrees of the second node in each link
        
    p = 1 - comb(M - kb, ka) / comb(M, ka)
    # Set probabilities that are zero to a small number to avoid -inf when
    # we take the log
    p[p == 0] = 1e-15
    assert (p > 0).all() and (p <= 1).all()
    if plot:
        _ = plt.hist(p, bins=100)
        plt.show()
        
    return p


def proba_comb_deg(x, links, verbose=False, plot=False):
    """Computes the probability of the given links, given that we know
    the number of nodes in the network and the degree pair.
    """
    p = np.zeros(links.shape[0])
    Na, Nb = x.shape # number of nodes 
    deg_a = np.sum(x, axis=1)
    deg_b = np.sum(x, axis=0)
    
    ka = deg_a[links[:,0]] # degrees of the first node in each link
    kb = deg_b[links[:,1]] # degrees of the second node in each link
    # Mask to avoid dividing by zero. When ka or kb is zero the limit of the 
    # probability is zero so this is fine, it just avoids numpy complaining
    # because it can't take limits
    mask_ab = np.logical_and(ka > 0, kb > 0)
    
    # Factor unfavorable cases/favorable cases
    uf = (Nb-ka[mask_ab])*(Na-kb[mask_ab])/(ka[mask_ab]*kb[mask_ab])
    # Don't forget to apply the mask here as well
    p[mask_ab] = 1 / (1 + uf)
    # Set probabilities that are zero to a small number to avoid -inf when
    # we take the log
    p[p == 0] = 1e-15
    assert (p > 0).all() and (p <= 1).all()
    if plot:
        _ = plt.hist(p, bins=100)
        plt.show()
        
    return p


def proba_comb_links_all(x, plot=False):
    """Computes proba_comb_links for all links in x and returns
    a probability matrix instead of a list.
    """
    M = np.sum(x)
    deg_a = np.sum(x, axis=1)
    deg_b = np.sum(x, axis=0)
    
    # Love this trick, see the docs for details
    ka, kb = np.meshgrid(deg_a, deg_b, indexing='ij')
        
    p = 1 - comb(M - kb, ka) / comb(M, ka)

    p[p == 0] = 1e-15
    assert (p > 0).all() and (p <= 1).all()
    if plot:
        _ = plt.hist(p, bins=100)
        plt.show()
        
    return p


def proba_comb_deg_all(x, plot=False):
    """Computes proba_comb_deg for all links in x and returns
    a probability matrix instead of a list.
    """
    p = np.zeros(x.shape)
    Na, Nb = x.shape 
    deg_a = np.sum(x, axis=1)
    deg_b = np.sum(x, axis=0)
    
    ka, kb = np.meshgrid(deg_a, deg_b, indexing='ij')
    mask_ab = np.logical_and(ka > 0, kb > 0)
    
    uf = (Nb-ka[mask_ab])*(Na-kb[mask_ab])/(ka[mask_ab]*kb[mask_ab])
    
    p[mask_ab] = 1 / (1 + uf)
        
    p[p == 0] = 1e-15
    assert (p > 0).all() and (p <= 1).all()
    if plot:
        _ = plt.hist(p, bins=100)
        plt.show()
        
    return p