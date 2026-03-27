#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 15:54:53 2019

@author: virati
Network Action - Compare ONT vs OFFT for EEG
"""

from DBSpace.control import proc_dEEG
import DBSpace as dbo
from DBSpace.visualizations import EEG_Viz
from DBSpace.control.TVB_DTI import DTI_support_model, plot_support_model

import cmocean
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('paper')
sns.set(font_scale=3)
sns.set_style('white')
#%%
pt_list = ['906','907','908']
do_condits = ['OnT','OffT']

## Basic initialization methods, need to suppress figures from these and clean these up
eFrame = proc_dEEG.proc_dEEG(pts=pt_list,procsteps='conservative',condits=do_condits)
eFrame.standard_pipeline()

#%% PSD plotting
eFrame.plot_psd(pt='907',condit='OnT',epoch='BONT')#'Off_3')

#%%
# Channel-marginalized Response Histogram
for pt in pt_list:
    eFrame.pop_meds(response=True,pt=pt)
    #eFrame.band_distr(do_moment='mads')
    plt.suptitle(pt)

#%%
# Here we'll plot the spatial distributions of \alpha
#for pt in pt_list:
   
#    eFrame.topo_median_response(do_condits=do_condits,pt=pt,band='Beta*')
    
#%%
#eFrame.topo_median_response(do_condits=do_condits,pt='POOL',band='Alpha',use_maya=False)