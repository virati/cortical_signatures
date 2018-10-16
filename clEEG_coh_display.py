#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 11:48:43 2018

@author: virati
Display script for coherence measures and etc.
"""

import sys
sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/')
import DBS_Osc as dbo

import matplotlib.pyplot as plt
import pickle
import numpy as np
import cmocean

from sklearn.decomposition import PCA, SparsePCA
import string

#%%
# Here, we're going to focus on Alpha only
pt_list = ['908']
condit = 'OnT'
epochs = ['Off_3','B' + condit.upper()]

coh_matrix = {epoch:[] for epoch in epochs}
for pt in pt_list:
    with open('/home/virati/Dropbox/Data/DBS'+pt+'_coh_dict2.pickle','rb') as handle:
        import_dict = pickle.load(handle)
        
        csd_dict = import_dict['CSD']
        plv_dict = import_dict['PLV']

    for epoch in epochs:
        band_ms_coh_matrix = np.zeros((256,256))
        band_phase_coh_matrix = np.zeros((256,256))
        for ii in range(256):
            for jj in range(ii):
                band_ms_coh_matrix[ii,jj] = plv_dict[pt][condit][epoch][ii][jj][0]
                band_phase_coh_matrix[jj,ii] = np.angle(csd_dict[pt][condit][epoch][ii][jj][0])
                band_phase_coh_matrix[ii,jj] = np.angle(csd_dict[pt][condit][epoch][ii][jj][0])
                
        band_ms_coh_matrix = band_ms_coh_matrix
        #band_ms_coh_matrix = band_ms_coh_matrix
        coh_matrix[epoch] = {'ms':band_ms_coh_matrix,'phase':band_phase_coh_matrix}
        
    plt.figure()
    plt.subplot(121)
    plt.pcolormesh(np.arange(256),np.arange(256),coh_matrix['BONT']['ms'] - coh_matrix['Off_3']['ms'])#,vmin=-np.pi,vmax=np.pi)
    plt.colorbar()
    plt.subplot(122)
    plt.pcolormesh(np.arange(256),np.arange(256),coh_matrix['Off_3']['phase'],cmap=cmocean.cm.phase)
    plt.colorbar()
    
    plt.figure()
    plt.hist(coh_matrix['Off_3']['phase'])
    
# PCA ROUTINE
#    pca = SparsePCA()
#    pca.fit(coh_matrix['BONT']['phase'])
#    plt.figure()
#    plt.subplot(211)
#    plt.pcolormesh(np.arange(256),np.arange(256),pca.components_)
#    plt.subplot(212)
#    plt.plot(pca.explained_variance_ratio_)
    
def plot_diff(condits):
    for condit in condits:
        plt.figure()
        plt.subplot(121)
        plt.pcolormesh(np.arange(256),np.arange(256),band_ms_coh_matrix)#,vmin=-np.pi,vmax=np.pi)
        plt.colorbar()
        plt.subplot(122)
        plt.pcolormesh(np.arange(256),np.arange(256),band_phase_coh_matrix,cmap=cmocean.cm.phase)
        plt.colorbar()
        #plt.imshow(band_ms_coh_matrix)
        plt.colorbar()
        plt.title(pt + condit + epoch)
        
def plot_epochs(epochs,condit='OnT'):
    ## Display part        
    plt.figure()
    plt.subplot(121)
    plt.pcolormesh(np.arange(256),np.arange(256),band_ms_coh_matrix)#,vmin=-np.pi,vmax=np.pi)
    plt.colorbar()
    plt.subplot(122)
    plt.pcolormesh(np.arange(256),np.arange(256),band_phase_coh_matrix,cmap=cmocean.cm.phase)
    plt.colorbar()
    #plt.imshow(band_ms_coh_matrix)
    plt.colorbar()
    plt.title(pt + condit + epoch)