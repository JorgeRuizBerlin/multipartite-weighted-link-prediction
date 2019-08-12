#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 10:53:16 2018

@author: paulh
"""
import numpy as np
import time
from scipy.special import comb


def bipartite_lcp_all(x, links, verbose=True):
    """Computes all LCP metrics for the bipartite case."""
    
    s = links.shape[0]
    init = np.zeros(s)
    
    lcl = np.zeros_like(init)
    cn = np.zeros_like(init)
    jc = np.zeros_like(init)
    car = np.zeros_like(init)
    cjc = np.zeros_like(init)
    cpa = np.zeros_like(init)
    ra = np.zeros_like(init)
    aa = np.zeros_like(init)
    cra = np.zeros_like(init)
    caa = np.zeros_like(init)
#     cpi = np.zeros_like(init)
    
    ne_r = dict()
    ne_c = dict()
    
    nodes_r = np.unique(links[:,0])
    nodes_c = np.unique(links[:,1])
    
    for i in range(x.shape[0]):
        if any(i==nodes_r):
            ne_c[i] = np.argwhere(x[i,:])
            
    for i in range(x.shape[1]):
        if any(i==nodes_c):
            ne_r[i] = np.argwhere(x[:,i])
            
    deg_r_all = np.sum(x, axis=1)
    deg_c_all = np.sum(x, axis=0)
    
    if verbose:
        print('Computing bipartite LCP')
        start = time.time()
        
    for i in range(s):
        
        n_r = ne_r[links[i,1]].flatten()
        n_c = ne_c[links[i,0]].flatten()
        if x[links[i,0], links[i,1]]:
            n_r = n_r[n_r != links[i,0]]
            n_c = n_c[n_c != links[i,1]]
            
        sx = x[n_r][:,n_c]
        lcl[i] = np.sum(sx)
        
        if lcl[i] > 0:
            
            ff_r = np.sum(sx, axis=1)
            ff_c = np.sum(sx, axis=0)
            
            cn_r = np.sum(ff_r > 0)
            cn_c = np.sum(ff_c > 0)
            
            cn[i] = cn_r + cn_c
            
            deg_r = deg_r_all[links[i,0]]
            deg_c = deg_c_all[links[i,1]]
            
            ext_r = deg_r - cn_r
            ext_c = deg_c - cn_c
            
            new_n_r = n_r[ff_r > 0]
            new_n_c = n_c[ff_c > 0]
            
            degree_rnnr = deg_r_all[new_n_r]
            degree_cnnc = deg_c_all[new_n_c]
            a = np.hstack((degree_rnnr,degree_cnnc))
            
            new_ff_r = ff_r[ff_r > 0]
            new_ff_c = ff_c[ff_c > 0]
            ff = np.hstack((new_ff_r,new_ff_c))
            
            jc[i] = cn[i]/(deg_r + deg_c)
            car[i] = cn[i]*lcl[i]
            cjc[i] = car[i]/(deg_r + deg_c)
            cpa[i] = ext_r*ext_c + ext_r*car[i] + ext_c*car[i] + car[i]*car[i]
            ra[i]  = np.sum(1/a)
            aa[i]  = np.sum(1/np.log2(a))
            cra[i] = np.sum(ff/a)
            caa[i] = np.sum(ff/np.log2(a))
#             fav = comb(x.shape[1]-1, deg_r-1) * comb(x.shape[0]-1, deg_c-1)
#             unfav = comb(x.shape[1]-1, deg_r) * comb(x.shape[0]-1, deg_c)
#             cpi[i] = np.log2(fav / (fav + unfav))
            
        if verbose:
            if i > 0 and i % (s//100) == 0:
                print('\r{:.2f} %, {:.2f} minutes'.format(i/s*100, 
                      (time.time()-start)/60), end='')
                
    if verbose:
        print('')
        
    return np.stack((lcl, cn, jc, car, cjc, cpa, ra, aa, cra, caa, cpi))


def bipartite_lcp_single(x, links, metric, verbose=True):
    """Computes a single bipartite LCP metric. Wildly inefficient 
    since it is just a wrapper to select the metric after computing 
    all the metrics."""
    
    metrics_list = ['lcl', 'cn', 'jc', 'car', 'cjc', 'cpa', 'ra', 'aa', 'cra', 'caa', 'cpi']
    assert metric in metrics_list
    lcp_scores = bipartite_lcp_all(x, links, verbose=verbose)
    
    return lcp_scores[metrics_list.index(metric)]


def tripartite_lcls(x, links, y, score=False):
    """Computes the LCLs in y of the given links in x."""
    s = links.shape[0]
    
    lcl_links = np.zeros(s, dtype=object)
    
    nodes_class1 = np.unique(links[:,0])
    nodes_class2 = np.unique(links[:,1])
    
    ne_class2 = dict()
    ne_class3 = dict()
    
    for i in range(x.shape[0]):
        if any(i == nodes_class1):
            ne_class2[i] = np.argwhere(x[i,:])
            
    for i in range(y.shape[0]):
        if any(i == nodes_class2):
            ne_class3[i] = np.argwhere(y[i,:])
    
    for i in range(s):
        
        n_class2 = ne_class2[links[i,0]].flatten()
        n_class3 = ne_class3[links[i,1]].flatten()
        
        if x[links[i,0], links[i,1]]:
            n_class2 = n_class2[n_class2 != links[i,0]]
            n_class3 = n_class3[n_class3 != links[i,1]]
        
        sy = y[n_class2][:,n_class3]
        
        lcl_links[i] = np.array([[n_class2[el[0]], n_class3[el[1]]] for el in 
                                np.argwhere(sy == 1)])
    
    if score:
        return [a.shape[0] for a in lcl_links]
    else:
        return lcl_links
    