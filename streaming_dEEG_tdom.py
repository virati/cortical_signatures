#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 12-04-2018

@author: virati
dEEG Continuous

Time domain analysis of dEEG data; simple ICA stuff for now

"""

from stream_dEEG import streamEEG


import sys
sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/')
import DBSpace as dbo
from DBSpace import nestdict

import itertools
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import numpy as np

import pickle

#%%

pts = ['906','907','908','910']
condits =  ['OnT','OffT']
baseline_calibrate = False
local_reref = False


#%%

perf_dict = nestdict()

class_type = 'l2'
do_classif = False


#%%
pt_test = [None] * len(pts)
pt_test_labels = [None] * len(pts)
pt_test_times = [None] * len(pts)

seg_labels = plt.figure()


# Iterate through our patients and set up the data

for pp,pt in enumerate(pts):
    pt_test[pp] = []
    pt_test_labels[pp] = []
    pt_test_times[pp] = []
    
    for condit in condits:
        print('Doing ' + pt + ' ' + condit)
        sEEG = streamEEG(ds_fact=2,pt=pt,condit=condit,spotcheck=True,do_L_reref=local_reref)
        
