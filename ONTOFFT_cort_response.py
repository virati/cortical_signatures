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

pt_list = ['906','907','908']

## Basic initialization methods, need to suppress figures from these and clean these up
eFrame = proc_dEEG.proc_dEEG(pts=pt_list,procsteps='conservative',condits=['OnT','OffT'])
eFrame.standard_pipeline()
#%%
# Channel-marginalized Responses
eFrame.pop_meds(response=True)

#%%
eFrame.band_distr()


#%%
# Here we'll plot the spatial distributions of \alpha
eFrame.topo_median_response()
