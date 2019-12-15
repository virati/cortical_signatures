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

import itertools
from itertools import product as cart_prod

from stream_dEEG import streamLFP

import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

import copy
from copy import deepcopy

class local_response:
    #Setup our main variables for the analysis
    TF_response = nestdict()
    Osc_response = nestdict()
    Osc_prebilat = nestdict()
    Osc_baseline = nestdict()
    
    #Ancillary analysis variables
    Osc_response_uncorr = nestdict()
    
    def __init__(self,analysis_windows=['Bilat','PreBilat'],do_pts = dbo.all_pts):
        # Which two epochs are we analysing?

        self.win_list = analysis_windows

        self.do_pts = do_pts
        
        self.colors = ['b','g']

        
    def extract_baselines(self):
        TF_response = self.TF_response
        Osc_response_uncorr = self.Osc_response_uncorr
        Osc_prebilat = self.Osc_prebilat
        Osc_baseline = self.Osc_baseline
        
        for pt,condit in cart_prod(self.do_pts,['OnT','OffT']):
            eg_rec = streamLFP(pt=pt,condit=condit)
            rec = eg_rec.time_series(epoch_name='PreBilat')
            TF_response[pt][condit] = eg_rec.tf_transform(epoch_name='Bilat')
            Osc_response_uncorr[pt][condit] = eg_rec.osc_transform(epoch_name='Bilat')
            
            Osc_prebilat[pt][condit] = eg_rec.osc_transform(epoch_name='PreBilat')
            #Find the mean within the prebilat for both left and right
            Osc_baseline[pt][condit] = [np.mean(Osc_prebilat[pt][condit][chann],axis=0) for chann in ['Left','Right']]
    
    def extract_response(self):
        Osc_response = deepcopy(self.Osc_response_uncorr)
        Osc_baseline = self.Osc_baseline
        
        for pt,condit in cart_prod(self.do_pts,['OnT','OffT']):
            for seg in range(self.Osc_response_uncorr[pt][condit]['Left'].shape[0]):
                for cc,chann in enumerate(['Left','Right']):
                    Osc_response[pt][condit][chann][seg,:] -= Osc_baseline[pt][condit][cc]
                    
        self.Osc_response = Osc_response
        
    def gen_osc_distr(self):
        Osc_response = self.Osc_response
        do_pts = self.do_pts
        
        
        self.Osc_indiv_marg = {pt:{condit:np.array((Osc_response[pt][condit]['Left'],Osc_response[pt][condit]['Right']))for condit in ['OnT','OffT']} for pt in do_pts}
        self.Osc_indiv_med = {pt:{condit:np.median(self.Osc_indiv_marg[pt][condit],axis=1) for condit in ['OnT','OffT']} for pt in do_pts}
        self.Osc_indiv_pop = {side:{condit:np.array([self.Osc_indiv_med[pt][condit][ss,:] for pt in do_pts]) for condit in ['OnT','OffT']} for ss,side in enumerate(['Left','Right'])}

    def plot_response(self):
        Osc_indiv_pop = self.Osc_indiv_pop
        color = self.colors
        
        for cc,chann in enumerate(['Left','Right']):
            plt.figure()
            ax2 = plt.subplot(111)
            distr = nestdict()
            for co,condit in enumerate(['OnT','OffT']):
                distr_to_plot = Osc_indiv_pop[chann][condit]
                
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
                
                plt.ylim((-6,30))
                distr[condit] = distr_to_plot
                    
            for bb in range(5):
                #rsres = stats.ks_2samp(distr['OnT'][:,bb],distr['OffT'][:,bb])
                rsres = stats.ranksums(distr['OnT'][:,bb],distr['OffT'][:,bb])
                #rsres = stats.ks_2samp(distr['OnT'][:,bb],distr['OffT'][:,bb])
                #rsres = stats.wilcoxon(distr['OnT'][:,bb],distr['OffT'][:,bb])
                #rsres = stats.ttest_ind(distr['OnT'][:,bb],distr['OffT'][:,bb])
                
                
                #ontres = stats.ranksums(distr['OnT'][:,bb])
                #ontres = stats.kstest(distr['OnT'][:,bb],cdf='norm')
                #ontres = stats.mannwhitneyu(distr['OnT'][:,bb])
                ontres = stats.ttest_1samp(distr['OnT'][:,bb],0)
                print(dbo.feat_order[bb])
                print(rsres)
                print(ontres)

#%%
analysis = local_response(do_pts = ['905','906','907','908'])
analysis.extract_baselines()
analysis.extract_response()
analysis.gen_osc_distr()
analysis.plot_response()


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
