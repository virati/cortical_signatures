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
import scipy.stats as stats

import pdb

import time
import pylab

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

def DEPRplot_flat_scalp(band,clims=(0,0),unwrap=True):
    #get coords
    egipos = mne.channels.read_montage('/home/virati/Dropbox/GSN-HydroCel-257.sfp')
    etrodes = egipos.pos
    
    cm = plt.cm.get_cmap('jet')
    
    if clims == (0,0):
        #clims = (np.min(band),np.max(band))
        clims = (0,257)
    
    #flatten coords
    flat_etrodes = np.copy(etrodes)
    flat_etrodes[:,2] = flat_etrodes[:,2] - np.max(flat_etrodes[:,2]) + 0.01
    
    if unwrap:
        flat_etrodes[:,0] = flat_etrodes[:,0] * -2*(flat_etrodes[:,2]) - 10*np.exp(flat_etrodes[:,2])
        flat_etrodes[:,1] = flat_etrodes[:,1] * -2*(flat_etrodes[:,2]) - 10*np.exp(flat_etrodes[:,2])
    
    plt.figure()
    sc = plt.scatter(flat_etrodes[:,0],flat_etrodes[:,1],c=np.arange(257),vmin=clims[0],vmax=clims[1],cmap=cm)
    plt.colorbar(sc)

def plot_3d_scalp(band,fig,n=1,clims=(0,0),label='generic',animate=False,unwrap=False,sparse_labels = True):
    #fig = plt.figure()
    
    egipos = mne.channels.read_montage('/home/virati/Dropbox/GSN-HydroCel-257.sfp')
    etrodes = egipos.pos
    
    #gotta normalize the color
    #band = np.tanh(band / 10) #5dB seems to be reasonable
    
    cm = plt.cm.get_cmap('jet')
    
    if clims == (0,0):
        clims = (np.min(band),np.max(band))
    
    if unwrap:
        flat_etrodes = np.copy(etrodes)
        flat_etrodes[:,2] = flat_etrodes[:,2] - np.max(flat_etrodes[:,2]) + 0.01
    
        flat_etrodes[:,0] = flat_etrodes[:,0] * -10*(flat_etrodes[:,2] + 3*1/(flat_etrodes[:,2] - 0.6) + 0.5)
        flat_etrodes[:,1] = flat_etrodes[:,1] * -10*(flat_etrodes[:,2] + 3*1/(flat_etrodes[:,2] - 0.6) + 0.5)
        
        ax = fig.add_subplot(1,1,n)
        sc = plt.scatter(flat_etrodes[:,0],flat_etrodes[:,1],c=band,vmin=clims[0],vmax=clims[1],s=300,cmap=cm,alpha=0.5)
        
        #Which channels are above two stds?
        zsc_band = stats.zscore(band)
        top_etrodes = np.where(np.abs(zsc_band) > 1)[0]
        
        if sparse_labels:
            annotate_list = top_etrodes
        else:
            annotate_list = range(257)
        
        for ii in annotate_list:
            plt.annotate('E'+str(ii+1),(flat_etrodes[ii,0],flat_etrodes[ii,1]),size=12)
        
        plt.axis('off')        
        
        plt.colorbar(sc)
        plt.title(label)
        
    else:
        ax = fig.add_subplot(1,1,n,projection='3d')
        sc = ax.scatter(etrodes[:,0],etrodes[:,1],10*etrodes[:,2],c=band,vmin=clims[0],vmax=clims[1],s=300,cmap=cm)
    
        try:plt.colorbar(sc)
        except: pdb.set_trace()
     
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
        
        ims = []
        plt.title(label)
        
        print('Animation: ' + str(animate))
        if animate:
            
            for angl in range(0,360,10):
                print('Animating frame ' + str(angl))
                ax.view_init(azim=angl)
                strangl = '000' + str(angl)
                plt.savefig('/tmp/'+ label + '_' + strangl[-3:] + '.png')
                time.sleep(.3)
    