#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:05:33 2017

@author: virati
Neighbor Montage methods
The sole purpose of this is to generate a laplacian for the channels in the EEG+LFP so we can do neighbor subtractions and re-ref for more local signals
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
import pdb

from scipy.sparse import dok_matrix

def return_cap_L(dth = 2.05,plot=False):
    #get the GSN channel information
    egipos = mne.channels.read_montage('/home/virati/Dropbox/GSN-HydroCel-257.sfp')
    e_locs = egipos.pos #this is shaped 257x3
    
    #compute distance matrix for all channels; this should be N x N
    nchann = e_locs.shape[0]
    dist_matr = np.zeros((nchann,nchann))
    
    for cc1 in range(nchann):
        for cc2 in range(nchann):
            if cc2 >= cc1:
                dist_matr[cc1,cc2] = np.linalg.norm(e_locs[cc1,:] - e_locs[cc2,:])
                
    dist_flat = dist_matr.flatten()
    
    dist_flat[dist_flat >= dth] = 0
    dist_matr = dist_flat.reshape(257,257)
    
    if plot:
        plt.figure()
        plt.subplot(211)
        plt.hist(dist_matr)
        plt.subplot(212)
        plt.imshow(dist_matr)
        plt.show()
    
    return dist_matr


#%%
#wrapper function for both neighbor/local montage and for PW-montage
    
def pw_montage(data,distr_matr):
    #How many edges does the distr_matrix have?
    np.linalg.norm(distr_matr,ord=0)
    pdb.set_trace()
    rrdata = np.zeros

#%%
#Function to actually FIND the Neighbor Montage from passed in data

def diff_reref(data,dist_matr):
    
    rrdata = [] #dok_matrix((257,257,data[0].shape[0]))
    flag_chann = np.zeros((257,257))
    
    for cc1 in range(257):
        neigh_vect = dist_matr[cc1,:]
        neigh_channs = np.where(neigh_vect != 0)[0]
        if neigh_channs.shape[0] != 0:
            #we're going to go to EVERY neighbor, compute the difference, and put it into a new
            for cc2 in neigh_channs:
                diff_chann = data[cc1] - data[cc2]
                rrdata.append((diff_chann,(cc1,cc2)))
                #[cc1][cc2] = diff_chann
                flag_chann[cc1,cc2] = 1
        else:
            rrdata.append((diff_chann,(cc1,cc1))) #[cc1] = data[cc1]
            flag_chann[cc1,cc1] = 1
                
    return rrdata
    
def reref_data(data,dist_matr,method='local'):
    #DATA needs to be an array corresponding to the full data stack
    # maybe....
    
    rrdata = np.zeros_like(data)
    flag_chann = np.zeros((257,1))
    
    for cc1 in range(257):
        neigh_vect = dist_matr[cc1,:]
        #Find the channels that are considered neighbors
        neigh_channs = np.where(neigh_vect != 0)
        if neigh_channs[0].shape[0] != 0:
            neigh_sum = np.zeros_like(data[0])
            for cc_n in neigh_channs:
                neigh_sum += data[cc_n][0,:]
            #meanify
            neigh_sum = 1/len(neigh_channs) * neigh_sum
            rrdata[cc1] = data[cc1] - neigh_sum
            flag_chann[cc1] = 0
        else:
            rrdata[cc1] = data[cc1]
            flag_chann[cc1] = 1
    
    print('Done finding Neighbor Montage...')
    
    return rrdata

#This gives us the dist_matrix we need


if __name__=='__main__':
    return_cap_L(dth=100,plot=True)