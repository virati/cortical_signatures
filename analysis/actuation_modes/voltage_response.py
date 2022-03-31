#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 15:32:53 2019

@author: virati
Script to load and characterize Voltage-sweep data; Most likely just from 906
"""

from dbspace.control.stream_buffers import streamEEG
from dbspace.utils.structures import nestdict

import itertools
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import numpy as np 

import pickle

from dbspace.viz.MM import EEG_Viz

#%%
#First, let's bring in the timeseries from DBS906 Voltage Sweep Experiment
pt = '906'
#%%
sEEG = streamEEG(ds_fact=2,pt=pt,condit='Volt',spotcheck=True,reref_class='none')
#%%
sEEG.seg_PSDs()
if pt == '906':
    blintv = (20,40)
elif pt == '907':
    blintv = (0,20)
    
sEEG.calc_baseline(intv=blintv)
#%%
sEEG.label_segments()
sEEG.plot_segment_labels()


#%%
# This is for 906
if pt == '906':
    intv_list = {0:(20,40),2:(45,55),3:(65,75),4:(84,94),5:(104,114),6:(125,135),7:(144,154)}
elif pt == '907':
    intv_list = {0:(0,20),2:(25,36),3:(45,56),4:(65,76),5:(85,96),6:(105,116),7:(125,136)}
    
intv_order = [0,2,3,4,5,6,7]

median_response = []
for voltage in intv_order:
    
    median_response.append(sEEG.median_response(intv=intv_list[voltage],do_plot=True))
    
    plt.suptitle(voltage)
    
#%%
median_response = np.array(median_response)
EEG_Viz.plot_3d_scalp(np.mean(median_response,axis=0)[:,2],unwrap=True)
#%%
#Need to go into each channel and figure out which ones "GROW" and which ones don't
from scipy.stats import linregress

def isMonotonic(A): 
  
    return (all(A[i] <= A[i + 1] for i in range(len(A) - 1)) or
            all(A[i] >= A[i + 1] for i in range(len(A) - 1))) 

chann_mono = np.zeros((257,))
chann_monoish = np.zeros((257,))

for cc in range(257):
    voltage_diff = np.diff(median_response[:,cc,2])
    chann_mono[cc] = (voltage_diff > 0).all()
    chann_monoish[cc] = np.sum(voltage_diff > 0) > 4
    
EEG_Viz.plot_3d_scalp(chann_monoish,unwrap=True)