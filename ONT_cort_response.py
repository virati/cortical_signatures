#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 16:13:54 2018

@author: virati
Simple script that runs the jackknifing on EEG data
"""

from proc_dEEG import proc_dEEG
import DBSpace as dbo
from DBSpace.visualizations import EEG_Viz

import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('paper')
sns.set(font_scale=3)
sns.set_style('white')

import pickle
import cmocean
#%%
pt_list = ['905','906','907','908']
#The feature vector, in this case the frequencies
fvect = np.linspace(0,500,513)
do_coherence = False

## Basic initialization methods, need to suppress figures from these and clean these up
eFrame = proc_dEEG(pts=pt_list,procsteps='liberal',condits=['OnT'])
eFrame.extract_feats(polyorder=0)
#%%
eFrame.pool_patients()
band = 'Beta*'

for band in ['Theta','Alpha','Beta*']:
    for pt in ['908']:
        mean_response = eFrame.med_stats(pt=pt)
        fig = plt.figure()
      
        band_i = dbo.feat_order.index(band)
        EEG_Viz.plot_3d_scalp(mean_response['OnT'][:,band_i],fig,label='OnT Mean Response ' + band,unwrap=True)
#%%
for pt in ['908']:
    eFrame.OnT_response()