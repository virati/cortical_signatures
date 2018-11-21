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
from DBSpace import nestdict

import matplotlib.pyplot as plt
import pickle
import numpy as np
import cmocean

from sklearn.decomposition import PCA, SparsePCA
import string

import umap
import pdb
import cmocean

import sys
sys.path.append('/home/virati/Dropbox/projects/libs/robust-pca/')
import r_pca

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
with open('/home/virati/big_coher_matrix.pickle','rb') as handle:
    import_dict = pickle.load(handle)
csd_dict = import_dict['CSD']
plv_dict = import_dict['PLV']
#%% Reshape the inputs into matrices for each patient x condition
band_idx = 2
msCoh = nestdict()

coh_stack = nestdict()
for pt in pt_list:
    hist_plots = plt.figure()
    conn_plots = plt.figure()
    
    for cc,condit in enumerate(condit_list):
        csd_matrix = {'Off_3':[], clabel[condit]:[]}
        for epoch in ['Off_3',clabel[condit]]:
            csd_matrix[epoch] = np.swapaxes(np.array([[csd_dict[pt][condit][epoch][ii][jj] for jj in range(257)] for ii in range(257)]),2,3)
        
        # Actually, let's return the baseline (Off_3) median subtracted versions of all stim-conditions
        for ss in range(csd_matrix[clabel[condit]].shape[3]):
            csd_matrix[clabel[condit]][:,:,:,ss] = csd_matrix[clabel[condit]][:,:,:,ss] - np.median(csd_matrix['Off_3'],axis=-1)
            
        coh_stack[pt][condit] = csd_matrix
        
        #lim_time = range()
        #plot the average through segments
        plt.figure(conn_plots.number)
        plt.subplot(2,2,2*cc+1)
        mag_diff = np.log10(np.median(np.abs(csd_matrix[clabel[condit]][:,:,:,band_idx]),axis=2)) - np.log10(np.median(np.abs(csd_matrix['Off_3'][:,:,:,band_idx]),axis=2))
        plt.imshow(mag_diff,vmax=0.5,vmin=-0.5)
        plt.colorbar()
        
        plt.subplot(2,2,2*(cc)+2)
        angle_diff = np.median(np.angle(csd_matrix[clabel[condit]][:,:,:,band_idx]),axis=2) - np.median(np.angle(csd_matrix['Off_3'][:,:,:,band_idx]),axis=2)
        
        angle_diff = (angle_diff + np.pi) % (2 * np.pi ) - np.pi
        
        plt.imshow(angle_diff,cmap=cmocean.cm.phase)
        plt.colorbar()
        plt.title(pt + ' ' + condit + ' difference')
        
        # TODO do masks so we focus just on the most important channels
        
        ## Plot histograms now
        plt.figure(hist_plots.number)
        plt.subplot(2,2,3)
        plt.hist(mag_diff.flatten(),bins=np.linspace(-0.5,0.5,20),alpha=0.4,label=condit)
        plt.ylim((0,256*256/2))
        plt.legend()
    
        plt.subplot(2,2,4)
        plt.hist(angle_diff.flatten(),alpha=0.4)
        plt.ylim((0,256*256/2))
        plt.legend()
        
        # Do a mask
        
        msCoh[pt][condit] = mag_diff
        
        
#%%
#work directly with the full stack across patients
# rPCA
band_idx = 1
coh_tensor = np.concatenate([np.abs((coh_stack[pt][condit][epoch])) for pt in pt_list for condit in condit_list for epoch in [clabel[condit]]],axis=3)[:,:,band_idx,:]
#coh_tensor = np.append([[([np.swapaxes(coh_stack[pt][condit][epoch],3,2) for epoch in ['Off_3',clabel[condit]]]) for condit in condit_list] for pt in pt_list],axis=3)
#%%
def do_pca():
    rpca = r_pca.R_pca(coh_tensor)
    L,S = rpca.fit()



#%%
# UMAP
def do_umap():
    import umap
    udim = umap.UMAP().fit_transform(coh_tensor.reshape(257*257,-1))
    return udim
udim = do_umap()
#%%
def plot_umap(udim):
    wrap_chann_udim = udim.reshape(257,257,2)
    use_map = matplotlib.cm.get_cmap('Spectral')
    plt.figure()
    for ii in range(257):
        sc = plt.scatter(wrap_chann_udim[ii,ii:,0],wrap_chann_udim[ii,ii:,1],color=use_map(ii/257),alpha=0.1,cmap=use_map)
    plt.colorbar(sc)
plot_umap(udim)
#%%
stat_coh = nestdict()
var_coh = nestdict()
for condit in condit_list:
    stat_coh[condit] = np.mean([coh[condit] for pt,coh in msCoh.items()],axis=0)
    var_coh[condit] = np.std([coh[condit] for pt,coh in msCoh.items()],axis=0)
    
    
plt.figure()
plt.subplot(1,2,1)
#plt.imshow(stat_coh['OnT'] / var_coh['OnT'],vmin=-100,vmax=100);plt.colorbar()
plt.hist((stat_coh['OnT']).flatten(),bins=np.linspace(-1,1,100))


plt.title('OnT')
plt.subplot(1,2,2)
#plt.imshow(stat_coh['OffT'] / var_coh['OffT'],vmin=-100,vmax=100);plt.colorbar()
plt.hist((stat_coh['OffT']).flatten(),bins=np.linspace(-1,1,100))

plt.title('OffT')
            
#%%
# Compare OnTarget to No Stim
band_idx=1
for pt in pt_list:
    bins = np.linspace(-4,4,100)
    plt.figure()
    plt.subplot(2,2,1)
    csd_matrix = np.array([[csd_dict[pt]['OnT']['Off_3'][ii][jj] for jj in range(257)] for ii in range(257)])
    plt.imshow(np.log10(np.abs(np.mean(csd_matrix[:,:,:,band_idx],axis=2))))
    plt.colorbar()
    
    plt.subplot(2,1,2)
    plt.hist(np.log10(np.abs(np.mean(csd_matrix[:,:,:,band_idx],axis=2)).flatten()),alpha=0.2,bins=bins,label='OFF')
    
    plt.subplot(2,2,2)
    csd_matrix = np.array([[csd_dict[pt]['OnT']['BONT'][ii][jj] for jj in range(257)] for ii in range(257)])
    plt.imshow(np.log10(np.abs(np.mean(csd_matrix[:,:,:,band_idx],axis=2))))
    
    plt.subplot(2,1,2)
    plt.hist(np.log10(np.abs(np.mean(csd_matrix[:,:,:,band_idx],axis=2)).flatten()),alpha=0.2,bins=bins,label='BONT')
    plt.legend()
            

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