#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 12:50:12 2020

@author: virati
DO for W-C re-simulation of Li work
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import sys
sys.path.append('/home/virati/Dropbox/projects/Research/DBSControl/autoDyn/')
from dyn_sys import dyn_sys
import networkx as nx
import pdb


def sigmoid(x, a, thr):
    return 1 / (1 + np.exp(-a * (x - thr)))

# nonlinear functions￼
def Se(x):
    aE = 50
    thrE = 0.125
    return sigmoid(x, thrE, aE) - sigmoid(0, thrE, aE)

def Si(x):
    aI = 50
    thrI = 0.4
    return sigmoid(x, thrI, aI) - sigmoid(0, thrI, aI)


#%%
class hopfer(dyn_sys):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        
        self.params = kwargs['params']
    
        self.rE = 0
        self.rI = 0
        self.c1 = 1.5
        self.c2 = 1
        self.c3 = 0.25
        self.c4 = 1
        
        self.wee = 1
        self.wie = 1
        self.wei = 1.5
        self.wii = 0.25
        self.thrE = 0.125
        self.thrI = 0.4
        self.tauE = 1/10
        self.tauI = 0.720325/10
        self.ae = 50
        self.ai = 50
        
        self.fdyn = self.f_drift
        
    def f_drift(self,x,ext_e = 0):
        exog = 0
        x_change = np.zeros((self.N,2))
        for nn,node in enumerate(x):
            exog = np.dot(self.L[nn,:],x[:,0])
            #x_change[nn,0] = -node[0] + (1-self.rE * node[0]) * Se(self.c1 * node[0] - self.c2 * node[1] + exog)
            #x_change[nn,1] = -node[1] + (1-self.rI * node[1]) * Si(self.c3 * node[0] - self.c4 * node[1])
            x_change[nn,0] = -node[0] + (1/(1+np.exp(-self.ae*(node[0] * self.wee - node[1] * self.wei - self.thrE + ext_e))))/self.tauE
            x_change[nn,1] = -node[1] + (1/(1+np.exp(-self.ai*(node[0] * self.wie - node[1] * self.wii - self.thrI))))/self.tauI
        return x_change

    def measure(self,x):
        return x

#%%

def do_filt():
    main_filter = delay_filt()
    main_filter.design_oscillation()
    main_filter.simulate()
    
wc_p = {'none':0}

def construct_graph():
    G = nx.Graph()
    G.add_nodes_from(['L-SCC','R-SCC','L-Front','R-Front','L-Temp','R-Temp'])
    G.add_edges_from([('L-SCC','R-SCC'),('L-Front','R-Front'),('L-Front','L-Temp'),('L-SCC','L-Temp'),('R-Front','R-Temp'),('R-SCC','R-Temp')])

    return nx.laplacian_matrix(G).todense()

if 0:
    SCCwm = construct_graph()
    main_sys = hopfer(N=6,L = SCCwm,params=wc_p)
    #main_sys.run(x_i = np.random.normal(0,1,size=(6,2)))
    main_sys.sim(x_i = np.random.normal(0,1,size=(6,2)))
    #%%
    main_sys.plot_measured()
