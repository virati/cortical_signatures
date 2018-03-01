#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 11:23:06 2017

@author: virati
This library is a small quick library for 3d plotting of EEG
"""

import matplotlib.pyplot as plt
import mne
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

def return_adj_net(dist_thresh = 3):
    egipos = mne.channels.read_montage('/tmp/GSN-HydroCel-257.sfp')
    etrodes = egipos.pos
    
    dist = np.zeros((257,257))
    for ii,ipos in enumerate(etrodes):
        #loop through all others and find the distances
        for jj,jpos in enumerate(etrodes):
            dist[ii][jj] = np.linalg.norm(ipos - jpos)
            
    mask = (dist <= dist_thresh).astype(int)
    
    return mask

def plot_3d_scalp(band,fig,n=1):
    #fig = plt.figure()
    ax = fig.add_subplot(1,1,n,projection='3d')
    egipos = mne.channels.read_montage('/home/virati/Dropbox/GSN-HydroCel-257.sfp')
    etrodes = egipos.pos
    
    #gotta normalize the color
    #band = np.tanh(band / 10) #5dB seems to be reasonable
    
    cm = plt.cm.get_cmap('jet')
    
    sc = ax.scatter(etrodes[:,0],etrodes[:,1],10*etrodes[:,2],c=band,vmin=np.min(band),vmax=np.max(band),s=300,cmap=cm)
    
    plt.colorbar(sc)
 
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # Get rid of the spines                         
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0)) 
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0)) 
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    #ax.xlim((-10,10))
    ax.set_xticks([])                               
    ax.set_yticks([])                               
    ax.set_zticks([])
    