#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 13:37:39 2020

@author: virati
Comb filtering example
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import sys
sys.path.append('/home/virati/Dropbox/projects/Research/DBSControl/autoDyn/')
from dyn_sys import dyn_sys
import networkx as nx

plt.close('all')


def sigmoid(x, a, thr):
    return 1 / (1 + np.exp(-a * (x - thr)))

# nonlinear functions
def Se(x):
    aE = 1.3
    thrE = 4
    return sigmoid(x, thrE, aE) - sigmoid(0, thrE, aE)

def Si(x):
    aI = 2
    thrI = 3.7
    return sigmoid(x, thrI, aI) - sigmoid(0, thrI, aI)

class delay_filt:
    def __init__(self,k=10):

        self.fs = 422
        self.tvect = np.linspace(0,60,60 * self.fs)
        self.k = k

    def design_oscillation(self):
        tvect = self.tvect
        center_freq = 10*np.exp(-0.1*tvect)#;plt.plot(center_freq)
        self.inp_sig = np.exp(-0.2*tvect)*(np.sin(2 * np.pi * center_freq * tvect) + 1/3 * np.sin(2 * np.pi * 3*center_freq * tvect) + 0/5 * np.sin(2 * np.pi * 5*center_freq * tvect))
        #inp_sig = np.exp(-0.2*tvect)*(sig.square(2*np.pi*center_freq * tvect)) #DO PAC OF THIS
        #inp_sig = np.exp(-0.1*tvect)*sig.chirp(tvect,f0=10,t1=10,f1=2,method='hyperbolic')
        #inp_sig[0:5] = 100
        self.y = np.zeros((self.inp_sig.shape[0] + 1,))
        
    def run(self):
        for tt,time in enumerate(self.tvect):
            self.y[tt] = self.inp_sig[tt] + 1*self.y[tt-self.k] + 0.01*np.random.normal(0,1)


    def simulate(self):
        self.run()
        
        tvect = self.tvect
        inp_sig = self.inp_sig
        y = self.y
        fs = self.fs
        
        #Plotting
        plt.figure()
        plt.plot(tvect,inp_sig)
        plt.plot(tvect,y[:inp_sig.shape[0]],'r')
        
        plt.figure()
        f,pxx = sig.welch(inp_sig,fs=fs,window='blackmanharris',nperseg=512,noverlap=500)
        f,pyy = sig.welch(y,fs=fs,window='blackmanharris',nperseg=512,noverlap=500)
        plt.plot(f,np.log10(pxx))
        plt.plot(f,np.log10(pyy),'r')

        plt.figure()
        nseg = 1024
        t,f,sgx = sig.spectrogram(inp_sig,fs=fs,window='blackmanharris',nperseg=nseg,noverlap=nseg-100,nfft=2**10)
        t,f,sgy = sig.spectrogram(y,fs=fs,window='blackmanharris',nperseg=nseg,noverlap=nseg-100,nfft=2**10)
        plt.pcolormesh(f,t,np.log10(sgy))
        plt.show()
        
class hopfer(dyn_sys):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        
        self.params = kwargs['params']
    
        self.rE = 1
        self.rI = 1
        self.c1 = 16
        self.c2 = 12
        self.c3 = 15
        self.c4 = 3
        
        self.fdyn = self.f_drift
        
    def f_drift(self,x):
        exog = 0
        x_change = np.zeros((self.N,2))
        for nn,node in enumerate(x):
            exog = np.dot(self.L[nn,:],x[:,0])
            x_change[nn,0] = -node[0] + (1-self.rE * node[0]) * Se(self.c1 * node[0] - self.c2 * node[1] + exog)
            x_change[nn,1] = -node[1] + (1-self.rI * node[1]) * Si(self.c3 * node[0] - self.c4 * node[1])
        
        return x_change

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

SCCwm = construct_graph()
main_sys = hopfer(N=6,L = SCCwm,params=wc_p)
main_sys.run(x_i = np.random.normal(0,1,size=(6,2)))