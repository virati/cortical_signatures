#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 20:25:09 2019

@author: virati
clClassif Script
"""

import DBSpace as dbo
from DBSpace.visualizations import EEG_Viz
import numpy as np


from proc_dEEG import proc_dEEG
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA, FastICA

sns.set_context('paper')
sns.set(font_scale=4)
sns.set_style('white')

all_pts = ['906','907','908']
#all_pts = ['906']

        
#UNIT TEST
EEG_analysis = proc_dEEG(pts=all_pts,procsteps='conservative',condits=['OnT','OffT'])
EEG_analysis.extract_feats(polyorder=0)
EEG_analysis.pool_patients_ONT()

#EEG_analysis.assess_binSVM()
#%%
EEG_analysis.train_binSVM(mask=False)
EEG_analysis.analyse_binSVM(approach='rpca')

#EEG_analysis.DEPRgen_OSC_stack()

#EEG_analysis.OBSsimple_stats()
# Binary Classification


#%%
EEG_analysis.OBSsimple_stats()

def cSVM(EEG_analysis):
    EEG_analysis.OBStrain_SVM(mask=False)
    fig = plt.figure();EEG_Viz.plot_3d_scalp(EEG_analysis.SVM_Mask.astype(np.int),fig)
    
    nomask_svm_coeff = EEG_analysis.SVM.coef_
cSVM(EEG_analysis)
