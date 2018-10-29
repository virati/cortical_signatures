#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 21:24:09 2018

@author: virati
dEEG Continuous
Load in continuous, raw dEEG from the mat converted files
This is the FIRST step in the streaming EEG pipeline

This file is essentially the file that loads in the raw EEG data, re-references, and presents the data in a format ready for pure_streamSVM to actually train and validate a classifier

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


#%%

perf_dict = nestdict()

class_type = 'l2'
do_classif = False
#%%
pt_test = [None] * len(pts)
pt_test_labels = [None] * len(pts)
pt_test_times = [None] * len(pts)

seg_labels = plt.figure()

for pp,pt in enumerate(pts):
    pt_test[pp] = []
    pt_test_labels[pp] = []
    pt_test_times[pp] = []
    
    for condit in condits:
        print('Doing ' + pt + ' ' + condit)
        sEEG = streamEEG(ds_fact=2,pt=pt,condit=condit,spotcheck=True)
        #sEEG.plot_TF(chann=32)
        #%%
        
        sEEG.seg_PSDs()
        
        #%%
        sEEG.calc_baseline()
        
        plt.figure()
        plt.plot(sEEG.true_labels,label=pt + condit)
        
        #%%
        if do_classif:
            #This CLASSIFIES the data
            perf_dict[pt][condit] = sEEG.classify_segs(ctype=class_type,train_type='stream')
        else:
            #This WRITES the data out:
            pt_test[pp].append(sEEG.gen_test_matrix())
            pt_test_labels[pp].append(sEEG.true_labels)
            pt_test_times[pp].append(sEEG.label_time)
            
#%%
with open('/tmp/big_file.pickle','wb') as file:
    pickle.dump({'States':pt_test,'Labels':pt_test_labels,'Times':pt_test_times},file)
    print('Successful Write of big pickle')    
