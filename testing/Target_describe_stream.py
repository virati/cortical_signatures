#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 15:32:53 2019

@author: virati
Script to load and characterize Voltage-sweep data; Most likely just from 906
"""

from DBSpace.control.stream_dEEG import streamEEG
import DBSpace as dbo
from DBSpace import nestdict

import itertools
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import numpy as np 

import pickle

from DBSpace.visualizations import EEG_Viz

#%%
#First, let's bring in the timeseries from DBS906 Voltage Sweep Experiment
pt = '906'
#%%
sEEG = streamEEG(ds_fact=2,pt=pt,condit='OnT',spotcheck=True,reref_class='none')
#%%
sEEG.seg_PSDs()
if pt == '906':
    blintv = (0,9)
elif pt == '907':
    blintv = (0,20)
    
sEEG.calc_baseline(intv=blintv)
#%%
sEEG.label_segments()
sEEG.plot_segment_labels()


#%%
# This is for 906
# Intervals are in SECONDS
if pt == '906':
    intv_list = {'L':(11,31),'R':(51,71),'BL':(91,111)}
elif pt == '907':
    intv_list = {'L':(11,31),'R':(51,71),'BL':(91,111)}
    
intv_order = ['L','R','BL']

median_response = []
for voltage in intv_order:
    
    median_response.append(sEEG.median_response(intv=intv_list[voltage],do_plot=True))
    
    plt.suptitle(voltage)
    
#%%
median_response = np.array(median_response)
EEG_Viz.plot_3d_scalp(np.mean(median_response,axis=0)[:,2],unwrap=True)
