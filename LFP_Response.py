#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 22:05:44 2018

@author: virati
LFP Response Script
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

analysis = network_action.local_response(do_pts = ['905','906','907','908'])
analysis.extract_baselines()
analysis.extract_response()
analysis.gen_osc_distr()
analysis.plot_response()

#%%
# BELOW NEEDS TO BE FOLDED INTO THE local_response CLASS


#%%
# here we'll work with the oscillatory state variables
Osc_pt_marg = {condit:np.array([(Osc_response[pt][condit]['Left'],Osc_response[pt][condit]['Right']) for pt in do_pts])for condit in ['OnT','OffT']}
Osc_pt_marg_bl = {condit:np.array([(Osc_prebilat[pt][condit]['Left'],Osc_prebilat[pt][condit]['Right']) for pt in do_pts])for condit in ['OnT','OffT']}
Osc_pt_marg_uncorr = {condit:np.array([(Osc_response_uncorr[pt][condit]['Left'],Osc_response_uncorr[pt][condit]['Right']) for pt in do_pts])for condit in ['OnT','OffT']}

#%%
# Do stats to see if the response is distributed the same as the no stim



#%%
for cc,chann in enumerate(['Left','Right']):
    #do violin plots
    fig = plt.figure()
    ax2 = plt.subplot(111)
    color = ['b','g']
    distr = nestdict()
    
    for co,condit in enumerate(['OnT','OffT']):
        #how many segments?
        #Here, we're going to plot ALL segments, marginalized across patients
        segNum = Osc_pt_marg[condit].shape[2]
        distr_to_plot = Osc_pt_marg[condit].swapaxes(1,2).reshape(len(do_pts)*segNum,2,5)[:,cc,:]
        parts = ax2.violinplot(distr_to_plot,positions=np.array([1,2,3,4,5]) + 0.2*co,showmedians=True)
        
        for partname in ('cbars','cmins','cmaxes','cmedians'):
            vp = parts[partname]
            vp.set_edgecolor(color[co])
            if partname == 'cmedians':
                vp.set_linewidth(5)
            else:
                vp.set_linewidth(2)
    
        for pc in parts['bodies']:
            pc.set_facecolor(color[co])
            pc.set_edgecolor(color[co])
            #pc.set_linecolor(color[co])
        
        distr[condit] = distr_to_plot
        #plt.plot([1,2,3,4,5],np.mean(distr_to_plot,axis=0),color=color[co])
    for bb in range(5):
        print(bb)
        #rsres = stats.ranksums(distr['OnT'][:,bb],distr['OffT'][:,bb])
        rsres = stats.ks_2samp(distr['OnT'][:,bb],distr['OffT'][:,bb])
        #rsres = stats.wilcoxon(distr['OnT'][:,bb],distr['OffT'][:,bb])
        #rsres = stats.ttest_ind(distr['OnT'][:,bb],distr['OffT'][:,bb])
        print(rsres)
        
        #ontres = stats.ranksums(distr['OnT'][:,bb])
        #ontres = stats.kstest(distr['OnT'][:,bb],cdf='norm')
        #ontres = stats.mannwhitneyu(distr['OnT'][:,bb])
        ontres = stats.ttest_1samp(distr['OnT'][:,bb],np.zeros((5,1)))
        print(ontres)
    
    plt.ylim((-30,50))
    plt.legend()
    
    #%%

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
