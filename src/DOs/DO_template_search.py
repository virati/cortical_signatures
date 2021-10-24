#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 23:07:30 2020

@author: virati
DO_Phase portrait and dynamics work
"""
import sys
sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/')
import DBSpace as dbo
import DBSpace.control.dyn_osc as dyn_osc
from DBSpace import nestdict
import pysindy as ps

from matplotlib import cm
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.signal as sig
from matplotlib.gridspec import GridSpec

#%%
Ephys = dyn_osc.Ephys

# Setup our chirp
pt = '906'
condit = 'OffTarget'

timeseries = dbo.load_BR_dict(Ephys[pt][condit]['Filename'],sec_offset=0)
end_time = timeseries['Left'].shape[0]/422

if pt == '903':
    tidxs = np.arange(231200,329300) #DBS903
if pt == '906':
    tidxs = np.arange(256000,330200) #DBS903

sos_lpf = sig.butter(10,10,output='sos',fs = 422)
filt_L = sig.sosfilt(sos_lpf,timeseries['Left']) 
#filt_L = sig.decimate(filt_L,2)[tidxs] #-211*60*8:
filt_R = sig.sosfilt(sos_lpf,timeseries['Right'])
#filt_R = sig.decimate(filt_R,2)[tidxs]

state = np.vstack((filt_L,filt_R))
sd = np.diff(state,axis=1,append=0)


#Let's take out the BL stim first from the raw timeseries
window = np.arange(255583,296095)
chirp = state[:,window]
#plt.figure()
#plt.plot(chirp.T)

## Now we get into subwindows
subwindow_e = np.array([0,470,3500,6350,9200,12300,30000])
chirplet = chirp[:,subwindow_e[1]:subwindow_e[1+1]]


#Ignore chirplet transform/analysis and just use the template to search amongst voltage sweep data
vsweep_fname = '/home/virati/MDD_Data/BR/905/Session_2015_09_02_Wednesday/Dbs905_2015_09_02_10_31_14__MR_0.txt'
vsweepData = dbo.load_BR_dict(vsweep_fname,sec_offset=0)

for side in ['Left','Right']:
    space_signal = vsweepData[side]
    space_signal = sig.detrend(space_signal)
    
    n_space = space_signal.shape[0]
    n_chirp = chirplet.shape[0]
    n_sweep = n_space - n_chirp
    
    tl_ip = np.zeros((n_sweep,2))
    
    for tlag in range(0,n_sweep,10):
        curr_sig = 

