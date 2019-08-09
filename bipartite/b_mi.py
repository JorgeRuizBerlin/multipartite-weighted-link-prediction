#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 11:36:13 2018

@author: paulh
"""
import numpy as np
import time
from b_lcp import tripartite_lcls


def mi_lcl_cc(x, links, y, p, verbose=False):
    
    mi = np.zeros(links.shape[0])
    lcl_links = tripartite_lcls(x, links, y)
    
    if verbose:
        print('Computing mutual information')
        start = time.time()
        
    for i in range(links.shape[0]):
        if lcl_links[i].any():
            cp = tripartite_cc(x, y, links[i], lcl_links[i])
            temp = -np.log2(p[links[i,0], links[i,1]]) + np.log2(cp[cp > 0])
            mi[i] = np.sum(temp)
            
            if verbose:
                if i > 0 and i % (links.shape[0]//100) == 0:
                    print('\r{:.2f} %, {:.2f} minutes'.format(i/links.shape[0]*100, 
                          (time.time()-start)/60), end='')
    
    if verbose:
        print('')
        
    return mi
    

def tripartite_cc(x, y, link_in_x, lcls_in_y):
    """Computes the conditional probability of a link in x 
    for each of its LCLs in y, using the clustering coefficient
    as the defined probability."""
    s = lcls_in_y.shape[0]
    # delete link in x if present, will affect the number of 
    # lcls of lcls
    flag_link_in_x = False
    if x[link_in_x[0], link_in_x[1]]:
        flag_link_in_x = True
        x[link_in_x[0], link_in_x[1]] = 0
        
    lcl_count = np.zeros(s) # numerator, number of actual quadrangles
    ne_prod = np.zeros(s) # denominator, number of possible quadrangles
    
    nodes_class2 = np.unique(lcls_in_y[:,0])
    nodes_class3 = np.unique(lcls_in_y[:,1])
    
    ne_class1 = dict() # neighbors in class 1 (ie neighbors of class 2 nodes)
    ne_class2 = dict() # neighbors in class 2 (ie neighbors of class 3 nodes)
    
    # iterate all class 2 nodes
    for i in range(y.shape[0]):
        # if the node is in our list of links
        if any(i == nodes_class2):
            # add its neighbors in class 1 to the dict
            ne_class1[i] = np.argwhere(x[:,i])
    
    # same for class 3 nodes
    for i in range(y.shape[1]):
        if any(i == nodes_class3):
            ne_class2[i] = np.argwhere(y[:,i])
    
    # iterate every given link
    for i in range(s):
        
        # list of neighbors for each node in the link.
        # since the link is between class 2 and 3
        # the neighbors we want will be class 1 and 2
        n_class1 = ne_class1[lcls_in_y[i,0]].flatten()
        n_class2 = ne_class2[lcls_in_y[i,1]].flatten()
        
        # even if the link under consideration is present, 
        # we don't want to consider the two nodes in it 
        # neighbors, so remove them from the lists
#        assert y[lcls_in_y[i,0], lcls_in_y[i,1]]
        n_class1 = n_class1[n_class1 != lcls_in_y[i,0]]
        n_class2 = n_class2[n_class2 != lcls_in_y[i,1]]
        
        # submatrix induced by neighbor sets
        sx = x[n_class1][:,n_class2]
        
        lcl_count[i] = np.sum(sx) 
        ne_prod[i] = n_class1.size * n_class2.size 
    
    p = (lcl_count / ne_prod)
    # remove nans from dividing zero by zero
    p[np.isnan(p)] = 0
    assert (p >= 0).all() and (p <= 1).all()
    
    # clean up
    if flag_link_in_x:
        x[link_in_x[0], link_in_x[1]] = 1
    
    return p


#def comb_mi(x, links, y, p):
#    
#    cp = proba_comb_3(x, links)
#    mi = -np.log2(p) + np.log2(cp)
#    
#    return mi