#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 16:13:54 2018

@author: virati
Simple script that runs the jackknifing on cleaned EEG data
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

pt_list = ['905','906','907','908']
#The feature vector, in this case the frequencies
fvect = np.linspace(0,500,513)
do_coherence = False

## Basic initialization methods, need to suppress figures from these and clean these up
eFrame = proc_dEEG(pts=pt_list,procsteps='conservative',condits=['OnT'])
eFrame.extract_feats(polyorder=0)

eFrame.pool_patients_ONT()
for pt in ['POOL']:
    eFrame.OnT_dr()

#%%
eFrame.pool_patients_ONT()
band = 'Beta*'

for band in ['Alpha','Beta*']:
    for pt in ['905','906','907','908','POOL']:
        mean_response = eFrame.med_stats(pt=pt,mfunc=np.median)
        
        #eFrame.support_analysis(pt=pt,band=band)
        
        fig = plt.figure()
      
        band_i = dbo.feat_order.index(band)
        EEG_Viz.plot_3d_scalp(mean_response['OnT'][:,band_i],fig,label='OnT Mean Response ' + band,unwrap=True)
        plt.suptitle(pt)
