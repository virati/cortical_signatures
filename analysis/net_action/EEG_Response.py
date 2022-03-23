#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 16:32:20 2018

@author: virati
Script to generate the EEG Response Violinplots

"""
#%%

from DBSpace.control import proc_dEEG
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

pt_list = ['906','907','908']

#The feature vector, in this case the frequencies
fvect = np.linspace(0,500,513)
do_coherence = False

## Basic initialization methods, need to suppress figures from these and clean these up
eFrame = proc_dEEG.proc_dEEG(pts=pt_list,procsteps='conservative',condits=['OnT','OffT'])
eFrame.standard_pipeline()

eFrame.band_distrs()
