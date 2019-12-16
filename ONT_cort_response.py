#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 16:13:54 2018

@author: virati
This scipt is focused on characterizing the ONTarget response
Includes some DTI support modeling which should be split out

"""

from DBSpace.control import proc_dEEG
import DBSpace as dbo
from DBSpace.visualizations import EEG_Viz
from DBSpace.control.TVB_DTI import DTI_support_model, plot_support_model


import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('paper')
sns.set(font_scale=3)
sns.set_style('white')

import pickle
import cmocean

pt_list = ['906','907','908']
## Basic initialization methods, need to suppress figures from these and clean these up
eFrame = proc_dEEG.proc_dEEG(pts=pt_list,procsteps='conservative',condits=['OnT'])
#%%
#eFrame.combined_bl()
#eFrame.combined_bl_distr(band='Alpha')
#%%
eFrame.OnT_dr(pt='POOL')
#%%
eFrame.plot_median_response(pt='POOL',use_maya=False)
#%%

#The feature vector, in this case the frequencies
fvect = np.linspace(0,500,513)
do_coherence = False

#%%
# Here we do the forward modeling to do network dissection
#eFrame.pool_patients()
for band in ['Alpha']:
    for pt in ['906']:
        #30, 25 is good
        EEG_support = DTI_support_model(pt,4,dti_parcel_thresh=15,eeg_thresh=55)
        plot_support_model(EEG_support,pt)
        eFrame.support_analysis(support_struct=EEG_support,pt=pt,band=band,voltage=str(3))
        
