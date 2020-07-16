#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 17:06:33 2019

@author: virati
Separate script just for control mode analysis of ONT (and OFFT)
"""


from DBSpace.control import proc_dEEG
import DBSpace as dbo
from DBSpace.visualizations import EEG_Viz
from DBSpace.control.TVB_DTI import DTI_support_model, plot_support_model

import seaborn as sns
import cmocean
sns.set_context('paper')
sns.set(font_scale=3)
sns.set_style('white')

## Basic initialization methods, need to suppress figures from these and clean these up
eFrame = proc_dEEG.proc_dEEG(pts=['906','907','908'],procsteps='conservative',condits=['OnT','OffT'])
eFrame.standard_pipeline()
#%%
## Let's plot all the bands first
for band in ['Alpha','Beta*']:
    eFrame.topo_median_response(do_condits=['OnT'],band=band,use_maya=True)

#%%
eFrame.topo_OnT_ctrl(pt='POOL',do_plot=True,plot_maya=True)

#%%

eFrame.topo_OnT_ctrl_tensor(pt='POOL')


#%%
# This one focuses on a single oscillatory band and tracks channels that 'change together'
eFrame.topo_OnT_alpha_ctrl(pt='POOL',do_plot=True,band='Alpha')
#Plot the \alpha specific control across time
eFrame.plot_alpha_ctrl_L(top_comp=5)
eFrame.plot_alpha_ctrl_S(top_comp=1)
#%%
#eFrame.control_rotate()