#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 13:32:10 2019

@author: virati
Script for analysis of *dynamics* during stimulation instead of response vector
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
import mayavi.mlab as mlab

import pickle
import cmocean


pt_list = ['906','907','908']
## Basic initialization methods, need to suppress figures from these and clean these up
eFrame = proc_dEEG.proc_dEEG(pts=pt_list,procsteps='liberal',condits=['OnT'])
#%%
eFrame.OnT_ctrl_dyn(do_plot=True)
