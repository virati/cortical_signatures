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

eFrame.OnT_ctrl_modes(pt='POOL')
#%%
eFrame.control_rotate()