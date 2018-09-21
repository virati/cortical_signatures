#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 16:32:20 2018

@author: virati
A REWRITE of the Cleaned EEG-Descriptive Pipeline (SCRIPT)
"""

#import sys
#sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/')
#import DBS_Osc as dbo

from proc_dEEG import proc_dEEG
from EEG_Viz import plot_3d_scalp

import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('paper')
sns.set(font_scale=4)
sns.set_style('white')

#%%
pt_list = ['906']

eFrame = proc_dEEG(pts=pt_list,procsteps='liberal',condits=['OnT','OffT'])
eFrame.extract_feats(polyorder=0)
#eFrame.gen_OSC_stack()

#%%
eFrame.band_stats(do_band='Alpha')

#%%
eFrame.interval_stats(do_band='Alpha')
#eFrame.psd_stats(chann_list=[])

#%%
## Do some coherence measures here

#coher_dict = eFrame.coher_stat(pt_list=pt_list,chann_list=[])
#fvect = np.linspace(0,500,513)
#%%

plt.figure()

#plt.plot(fvect,np.real(coher_dict['908']['OnT']['Off_3'][116][112][0]))
#plt.plot(fvect,np.imag(coher_dict['908']['OnT']['Off_3'][116][112][0]))

#%%
# Here, we're going to focus on Alpha only

condit = 'OffT'
if condit == 'OnT':
    epochs = ['Off_3','BONT']
else:
    epochs = ['BOFT']
coh_matrix = {epoch:[] for epoch in epochs}
for pt in pt_list:
    for epoch in epochs:
        band_ms_coh_matrix = np.zeros((256,256))
        band_phase_coh_matrix = np.zeros((256,256))
        for ii in range(256):
            for jj in range(ii):
                band_ms_coh_matrix[ii,jj] = (10*np.log10(np.median(np.abs(coher_dict[pt][condit][epoch][ii][jj][0]))) + 160)/50
                band_phase_coh_matrix[jj,ii] = np.median(np.angle(coher_dict[pt][condit][epoch][ii][jj][0]))
                
        #band_ms_coh_matrix = band_ms_coh_matrix
        coh_matrix[epoch] = {'ms':band_ms_coh_matrix,'phase':band_phase_coh_matrix}
        
        ## Display part
        
        plt.figure()
        plt.pcolormesh(np.arange(256),np.arange(256),band_phase_coh_matrix + band_ms_coh_matrix)#,vmin=-np.pi,vmax=np.pi)
        #plt.imshow(band_ms_coh_matrix)
        plt.colorbar()
        plt.title(pt + condit + epoch)