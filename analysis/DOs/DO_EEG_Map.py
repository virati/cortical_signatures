# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 12:18:37 2016

@author: virati
This script will try to import raw hdEEG data in a continuous way, preprocess it, and generate the figures needed for "Aim 2" - Mapping Cortical Responses/Signatures to Stimulation Parameters
THIS IS AN UPDATED FILE NOW SPECIFIC TO 906 until I fix the code to be modular/OOP
"""


import scipy
import scipy.io as sio
import scipy.signal as sig
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats
from numpy.random import default_rng

import mne
import pdb
import h5py

import pysindy as ps

from DBSpace.visualizations import EEG_Viz as EEG_Viz
from DBSpace.control.dyn_osc import EEG_DO

plt.rcParams['image.cmap'] = 'jet'
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 30}

matplotlib.rc('font', **font)
plt.close('all')
#%%

        
#%%
pt906_head = EEG_DO()

pt906_head.sgs()
#%%
pt906_head.map_blips(thresh=1)
pt906_head.phase(chs=[163,162],interval=(7400,8200),plot=True)


if 0:
    pt906_head.phase(chs=[255,256],interval=(7400,8200),plot=True)
    pt906_head.phase(chs=[255,256],interval=(9000,9800),plot=True)
    
    pt906_head.phase(chs=[32,256],interval=(9000,9800),plot=True)