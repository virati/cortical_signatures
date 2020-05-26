#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 22:05:44 2018

@author: virati
Network Action - Compare ONT vs OFFT for SCC-LFP
"""

import sys
sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/')
import DBSpace as dbo
from DBSpace import nestdict
from DBSpace.control import network_action

import itertools
from itertools import product as cart_prod

import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

import copy
from copy import deepcopy

do_pts = ['905','906','907','908']
analysis = network_action.local_response(do_pts = do_pts)
analysis.extract_baselines()
analysis.extract_response()
analysis.gen_osc_distr()

analysis.plot_responses(do_pts=do_pts)


#%%
#analysis.plot_patient_responses()
#%%

#analysis.plot_segment_responses(do_pts = do_pts)

#%%
# BELOW NEEDS TO BE FOLDED INTO THE local_response CLASS




#%%
# Do stats to see if the response is distributed the same as the no stim


    
    #%%
'''What am I doing here??'''
for cc in range(2):
    plt.figure()
    Osc_pt_median = {condit:np.mean(Osc_pt_marg[condit].swapaxes(1,2).reshape(len(do_pts)*segNum,2,5),axis=0) for condit in ['OnT','OffT']}
    
    plt.plot([1,2,3,4,5],(Osc_pt_median['OnT'][cc,:]).T,label='OnT',color=color[0])
    plt.plot([1,2,3,4,5],(Osc_pt_median['OffT'][cc,:]).T,label='OffT',color=color[1])
    plt.legend()
    
    
#%%
# This works directly with the PSDs
#TF_pt_marg[condit] = [ for pt,condit in cart_prod(dbo.all_pts,['OnT','OffT'])] 
TF_pt_marg = {condit:np.array([(TF_response[pt][condit]['Left']['SG'],TF_response[pt][condit]['Right']['SG']) for pt in do_pts])for condit in ['OnT','OffT']}

fvect = TF_response[pt][condit]['Left']['F']
#how many segments we got?
seg_num = TF_pt_marg['OnT'].shape[3]
plt.figure()
TF_pt_median = {condit:np.mean(TF_pt_marg[condit].swapaxes(1,3).reshape(6*seg_num,423,2),axis=0) for condit in ['OnT','OffT']}
plt.plot(fvect,10*np.log10(TF_pt_median['OnT'][:,0]))
plt.plot(fvect,10*np.log10(TF_pt_median['OffT'][:,0]))
