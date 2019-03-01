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

pts = ['905','906','907','908','910']
condits =  ['OnT','OffT']
baseline_calibrate = True
#local_reref = False
reref_class='local'

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
        sEEG = streamEEG(ds_fact=2,pt=pt,condit=condit,spotcheck=True,reref_class=reref_class,full_experiment=False)
        
        sEEG.seg_PSDs()
        
        #%%
        sEEG.calc_baseline(intv=(0,9))
        
        #This will display our experimental conditions, but from the presence of stimulation artifact
        #plt.figure()
        #plt.plot(sEEG.true_labels,label=pt + condit)
        
        #%%
        # We write the data to a dictionary with patient keys

        pt_test[pp].append(sEEG.gen_test_matrix())
        pt_test_labels[pp].append(sEEG.true_labels)
        pt_test_times[pp].append(sEEG.label_time)
            
#%%
#Finally, we pickle the dictionaries
        
with open('/tmp/big_file.pickle','wb') as file:
    pickle.dump({'States':pt_test,'Labels':pt_test_labels,'Times':pt_test_times},file)
    print('Successful Write of big pickle')    
