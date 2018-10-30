#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 11:48:43 2018

@author: virati
Display script for coherence measures and etc.
"""

#import sys
#sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/')
import DBSpace as dbo

import matplotlib.pyplot as plt
import pickle
import numpy as np
import cmocean

from sklearn.decomposition import PCA, SparsePCA
import string

import umap
import pdb


#%%
def umap_display():
    with open('/home/virati/Dropbox/Data/DBS'+pt+'_coh_dict.pickle','rb') as handle:
        import_dict = pickle.load(handle)
        
        csd_dict = import_dict['CSD']
        plv_dict = import_dict['PLV']
    



#%%
# Here, we're going to focus on Alpha only
pt_list = ['906','907','908']
condit_list = ['OnT','OffT']
clabel = {'OnT':'BONT','OffT':'BOFT'}
#epochs = ['Off_3',clabel[condit]]


#%% Reshape the inputs into matrices for each patient x condition
for pt in pt_list:
    for condit in condit_list:
        csd_matrix = {'Off_3':[], clabel[condit]:[]}
        for epoch in ['Off_3',clabel[condit]]:
            csd_matrix[epoch] = np.array([[csd_dict[pt][condit][epoch][ii][jj] for jj in range(257)] for ii in range(257)])
            
            
        #plot the average through segments
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(np.median(np.abs(csd_matrix[clabel[condit]][:,:,:,2]),axis=2) - np.median(np.abs(csd_matrix['Off_3'][:,:,:,2]),axis=2))
        plt.subplot(1,2,2)
        plt.imshow(np.median(np.angle(csd_matrix[clabel[condit]][:,:,:,2]),axis=2) - np.median(np.angle(csd_matrix['Off_3'][:,:,:,2]),axis=2))
        plt.colorbar()
        plt.suptitle(pt + ' ' + condit + ' difference')
            

#%%
if 0:
    for condit in ['OnT','OffT']:
        epochs = ['Off_3',clabel[condit]]
        coh_matrix = {epoch:[] for epoch in epochs}
        for pt in pt_list:
            with open('/home/virati/big_coher_matrix.pickle','rb') as handle:
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
                
            #pdb.set_trace()
            plt.figure()
            plt.subplot(121)
            diff_coh = coh_matrix[epochs[1]]['phase'] - coh_matrix[epochs[0]]['phase']
            plt.pcolormesh(np.arange(256),np.arange(256),diff_coh)#,vmin=-np.pi,vmax=np.pi)
            plt.colorbar()
            plt.subplot(122)
            plt.pcolormesh(np.arange(256),np.arange(256),coh_matrix['Off_3']['phase'],cmap=cmocean.cm.phase)
            plt.colorbar()
            
            udim = umap.UMAP().fit_transform(diff_coh)
            plt.figure()
            plt.scatter(udim[:,0],udim[:,1],c=np.linspace(0,256,256))#c=diff_coh)
            
            plt.figure()
            plt.hist(coh_matrix['Off_3']['phase'].flatten(),bins=100,label='NoStim',alpha=0.3)
            plt.hist(coh_matrix[clabel[condit]]['phase'].flatten(),bins=100,label='Stim',alpha=0.3)
            plt.legend()
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