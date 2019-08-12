#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 12:27:23 2018

@author: paulh
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx
from networkx.algorithms import bipartite


def giant_component_bp(x):
    x_sp = sp.bsr_matrix(x)
    G = bipartite.matrix.from_biadjacency_matrix(x_sp)
    giant = max(nx.connected_component_subgraphs(G), key=len)
    top_nodes = {n for n, d in giant.nodes(data=True) if d['bipartite']==0}
    bottom_nodes = set(giant) - top_nodes
    A = bipartite.matrix.biadjacency_matrix(giant, list(top_nodes), list(bottom_nodes))
    a = A.toarray()
    
    return a


def load_elegans_data():
    
    connectome = np.array(pd.read_csv('chemical_connectome.csv', header=None))
    nodes1 = np.array(pd.read_csv('Neurons-NT_allcategories.csv', header=None))
    nodes2 = np.array(pd.read_csv('NT-NR_allcategories.csv', header=None))
    nodes3 = np.array(pd.read_csv('NR-Neurons_wc_NEWORDER_mu.csv', header=None))
    
    # lists of the nodes in each class
    source_list = np.concatenate((connectome[:,0], nodes1[:,0]))
    source_list = np.unique(source_list)
    nt_list = np.concatenate(([x for l in nodes1[:,1] for x in l.split(',')], nodes2[:,0]))
    nt_list = np.unique(nt_list)
    nr_list = np.concatenate(([x for l in nodes3[:,0] for x in l.split(',')], nodes2[:,1]))
    nr_list = np.unique(nr_list)
    target_list = np.concatenate((connectome[:,1], nodes3[:,1]))
    target_list = np.unique(target_list)
    
    # dictionaries
    source_dict = {s:idx for idx, s in enumerate(source_list)}
    nt_dict = {nt:idx for idx, nt in enumerate(nt_list)}
    nr_dict = {nr:idx for idx, nr in enumerate(nr_list)}
    target_dict = {t:idx for idx, t in enumerate(target_list)}
    
    rev_source_dict = {value:key for key,value in source_dict.items()}
    rev_nt_dict = {value:key for key,value in nt_dict.items()}
    rev_nr_dict = {value:key for key,value in nr_dict.items()}
    rev_target_dict = {value:key for key,value in target_dict.items()}
    
    source_nt = np.zeros((source_list.size, nt_list.size), dtype=np.int32)
    for line in nodes1:
        i = source_dict[line[0]]
        for el in line[1].split(','):
            j = nt_dict[el]
            source_nt[i,j] = 1
    
    nt_nr = np.zeros((nt_list.size, nr_list.size), dtype=np.int32)
    for line in nodes2:
        i = nt_dict[line[0]]
        j = nr_dict[line[1]]
        nt_nr[i,j] = 1
    
    nr_target = np.zeros((nr_list.size, target_list.size,), dtype=np.int32)
    for line in nodes3:
        j = target_dict[line[1]]
        for el in line[0].split(','):
            i = nr_dict[el]
            nr_target[i, j] = 1
            
    chem_conn = np.zeros((source_list.size, target_list.size), dtype=np.int32)
    for line in connectome:
        i = source_dict[line[0]]
        j = target_dict[line[1]]
        chem_conn[i,j] = 1
        
    source_target = ((source_nt @ nt_nr @ nr_target) > 0).astype(np.int32)
    
    return source_nt, nt_nr, nr_target


def load_gpcr_data():
    data = np.array(pd.read_csv('bind_orfhsa_drug_gpcr.txt', delim_whitespace=True, header=None))

    sources = np.unique(data[:,0])
    targets = np.unique(data[:,1])
    
    source_dict = {s:idx for idx, s in enumerate(sources)}
    target_dict = {t:idx for idx, t in enumerate(targets)}
    rev_source_dict = {val:key for key, val in source_dict.items()}
    rev_target_dict = {val:key for key, val in target_dict.items()}
    
    x_g = np.zeros((sources.size, targets.size)).astype(np.int32)
    
    for line in data:
        i = source_dict[line[0]]
        j = target_dict[line[1]]
        x_g[i,j] = 1
        
    return x_g


def load_enzyme_data():
    data_e = np.array(pd.read_csv('bind_orfhsa_drug_e.txt', delim_whitespace=True, header=None))
    
    sources_e = np.unique(data_e[:,0])
    targets_e = np.unique(data_e[:,1])
    
    source_dict_e = {s:idx for idx, s in enumerate(sources_e)}
    target_dict_e = {t:idx for idx, t in enumerate(targets_e)}
    rev_source_dict_e = {val:key for key, val in source_dict_e.items()}
    rev_target_dict_e = {val:key for key, val in target_dict_e.items()}
    
    x_e = np.zeros((sources_e.size, targets_e.size)).astype(np.int32)
    
    for line in data_e:
        i = source_dict_e[line[0]]
        j = target_dict_e[line[1]]
        x_e[i,j] = 1
        
    return x_e

def load_ion_channel_data():
    data_ic = np.array(pd.read_csv('bind_orfhsa_drug_ic.txt', delim_whitespace=True, header=None))

    sources_ic = np.unique(data_ic[:,0])
    targets_ic = np.unique(data_ic[:,1])
    
    source_dict_ic = {s:idx for idx, s in enumerate(sources_ic)}
    target_dict_ic = {t:idx for idx, t in enumerate(targets_ic)}
    rev_source_dict_ic = {val:key for key, val in source_dict_ic.items()}
    rev_target_dict_ic = {val:key for key, val in target_dict_ic.items()}
    
    x_ic = np.zeros((sources_ic.size, targets_ic.size)).astype(np.int32)
    
    for line in data_ic:
        i = source_dict_ic[line[0]]
        j = target_dict_ic[line[1]]
        x_ic[i,j] = 1
        
    return x_ic

