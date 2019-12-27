#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 20:25:09 2019

@author: virati
clClassif Script
Binary Classification for Cleaned EEG Data
"""

import DBSpace as dbo
from DBSpace.visualizations import EEG_Viz
from DBSpace.control import proc_dEEG
import numpy as np


#from proc_dEEG import proc_dEEG
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA, FastICA

sns.set_context('paper')
sns.set(font_scale=4)
sns.set_style('white')

all_pts = ['906','907','908']
#all_pts = ['906']

EEG_analysis = proc_dEEG.proc_dEEG(pts=all_pts,procsteps='conservative',condits=['OnT','OffT'])
#%%
#x,y,z = EEG_analysis.get_SVM_dsgn(do_plot=True)

#%%

EEG_analysis.train_binSVM(mask=False)

#%%
#EEG_analysis.new_SVM_dsgn(do_plot=True)
#%%
EEG_analysis.oneshot_binSVM()
EEG_analysis.bootstrap_binSVM()
EEG_analysis.analyse_binSVM()

#EEG_analysis.OnT_dr(data_source=EEG_analysis.SVM_coeffs)
#%%
EEG_analysis.learning_binSVM()

