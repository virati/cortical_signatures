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

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('paper')
sns.set(font_scale=4)
sns.set_style('white')

#%%
pt_list = ['907']

eFrame = proc_dEEG(pts=pt_list,procsteps='liberal',condits=['OnT','OffT'])
eFrame.extract_feats(polyorder=0)
#eFrame.gen_OSC_stack()

#%%
#eFrame.band_stats(do_band='Theta')

#eFrame.psd_stats(chann_list=[])

#%%
# Do some coherence measures here
coher_dict = eFrame.coher_stat(chann_list=[])
#%%

plt.figure()
fvect = coher_dict['907']['OnT']['Off_3'][116][112][0]
plt.plot(fvect,np.real(coher_dict['907']['OnT']['Off_3'][116][112][1]))
plt.plot(fvect,np.imag(coher_dict['907']['OnT']['Off_3'][116][112][1]))

#%%
# Here, we're going to focus on Alpha only
alpha_idxs = np.where(np.logical_and(fvect <= 30, fvect >= 14))
band_coh_matrix = np.zeros((256,256))

condit = 'OnT'
if condit == 'OnT':
    epochs = ['Off_3','BONT']
else:
    epochs = ['Off_3','BOFT']
for pt in pt_list:
    for epoch in epochs:
        for ii in range(256):
            for jj in range(ii):
                band_coh_matrix[ii,jj] = np.median(np.real(coher_dict[pt][condit][epoch][ii][jj][1][alpha_idxs]))
                band_coh_matrix[jj,ii] = np.median(np.imag(coher_dict[pt][condit][epoch][ii][jj][1][alpha_idxs]))
                
        plt.figure()
        plt.pcolormesh(np.arange(256),np.arange(256),band_coh_matrix,vmin=-2,vmax=2)
        plt.colorbar()
        plt.title(pt + condit + epoch)