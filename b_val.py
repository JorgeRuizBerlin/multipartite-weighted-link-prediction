#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 10:20:30 2018

@author: paulh
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc


def get_links_to_del(x, fraction=0.1, loops=10):
    """Generates lists of links to delete from an adjacency matrix,
    one list for each loop in loops."""
    present = np.argwhere(x == 1)
    num_to_del = int(present.shape[0] * 0.1)
    idx_to_del = [
            np.random.choice(present.shape[0], size=num_to_del, replace=False) 
            for i in range(loops)]
    links_to_del = np.array([present[idx] for idx in idx_to_del])
    
    return links_to_del


def delete_links(x, links_to_del):
    """Deletes the specified links from an adjacency matrix."""
    x_ = np.copy(x)
    for link in links_to_del:
        x_[link[0], link[1]] = 0
        
    return x_


def evaluate_predictions(true, pred, plot=False):
    """Evaluates prediction quality using area under the PR 
    and ROC curves."""
    precision, recall, _ = precision_recall_curve(true, pred)
    fpr, tpr, _ = roc_curve(true, pred)
    aupr = auc(recall, precision)
    auroc = auc(fpr, tpr)
    
    if plot:
        fig, ax = plt.subplots()
        ax.plot(recall, precision, '.')
        ax.plot(fpr, tpr, '.')
        plt.show()
    
    return aupr, auroc


def cross_val(f, x, links_to_del, *args, y=None, mode='lcp', loops=10, 
              raw_output=False, verbose=True, plot=False):
    assert links_to_del.shape[0] == loops
    if raw_output:
        scores_list = []
        true_list = []
    else:
        results = np.zeros((loops, 2))

    for i in range(loops):
        if verbose:
            print('Trial {} of {}'.format(i + 1, loops))
        x_ = delete_links(x, links_to_del[i])
        missing = np.argwhere(x_ == 0)
        true = x[missing[:,0], missing[:,1]]
        
        if mode == 'lcp':
            score = f(x_, missing, *args)
        elif mode == 'lcp_tp':
            score = f(x_, missing, y, *args)
        elif mode == 'si_bp':
            score = f(x_, missing, x_.T, *args)
        elif mode == 'si_tp':
            score = f(x_, missing, y, *args)
        else:
            raise ValueError('Invalid scoring mode')
            
        if raw_output:
            scores_list.append(score)
            true_list.append(true)
        else:
            results[i] = evaluate_predictions(true, score, plot=plot)
        
    if raw_output:
        return true_list, scores_list
    else:
        return results


#def cross_val_si_bp(f, x, *args, loops=10, fraction=0.1, 
#              verbose=True, plot=False):
#    
#    results = np.zeros((loops, 2))
#    for i in range(loops):
#        if verbose:
#            print('Trial {} of {}'.format(i + 1, loops))
#        x_ = delete_bp_links(x, fraction)
#        missing = np.argwhere(x_ == 0)
#        true = x[missing[:,0], missing[:,1]]
#        score = f(x_, missing, x_.T, *args)
#        results[i] = evaluate_predictions(true, score, plot=plot)
#        
#    return results
#
#
#def cross_val_si_tp(f, x, y, *args, loops=10, fraction=0.1, 
#              verbose=True, plot=False):
#    
#    results = np.zeros((loops, 2))
#    for i in range(loops):
#        if verbose:
#            print('Trial {} of {}'.format(i + 1, loops))
#        x_ = delete_bp_links(x, fraction)
#        missing = np.argwhere(x_ == 0)
#        true = x[missing[:,0], missing[:,1]]
#        score = f(x_, missing, y, *args)
#        results[i] = evaluate_predictions(true, score, plot=plot)
#        
#    return results
