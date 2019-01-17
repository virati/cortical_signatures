#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 16:32:20 2018

@author: virati
This script is focused on capturing the difference between ONTarget and OFFTarget

"""

#import sys
#sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/')
#import DBS_Osc as dbo

from proc_dEEG import proc_dEEG
from DBSpace.visualizations import EEG_Viz

import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('paper')
sns.set(font_scale=3)
sns.set_style('white')

import pickle
import cmocean

pt_list = ['906','907','908']
#pt_list=['908']
#The feature vector, in this case the frequencies
fvect = np.linspace(0,500,513)
do_coherence = False

## Basic initialization methods, need to suppress figures from these and clean these up
eFrame = proc_dEEG(pts=pt_list,procsteps='conservative',condits=['OnT','OffT'])
#%%

eFrame.plot_median_response(condit='OnT',use_maya=False)
eFrame.plot_median_response(condit='OffT',use_maya=False)
#%%
eFrame.compute_response()
## This looks at the stats of the response
eFrame.response_stats(plot=True,band='Alpha')




#%%

'''
EVERYTHING BELOW needs to be cleaned up


'''

#%%
if 0:
    
    
    


    eFrame.DEPRgen_OSC_stack()
    #%%
    do_band = 'Alpha'
    
    #%%
    eFrame.simple_stats()
    eFrame.band_stats()
    #%%
    eFrame.band_distr()
    eFrame.plot_band_stats(do_band=do_band)

    
    #%%
    # Secondary analyses - Started Jan 2019
    # Find the combined baseline distribution
    eFrame.combined_bl()
    eFrame.compute_response()
    #If we want to plot all the violinplots for all channels; we rarely do
    #eFrame.combined_bl_distr(ch=10)
    
    #%%
    eFrame.per_chann_stats(condit='OnT',band=do_band)
    
    #%%
    eFrame.ONTvsOFFT(band=do_band,stim=0)
    
    
    
    #%%
    #
    eFrame.simple_stats()
    

    #
    eFrame.band_stats()
    #%%
    # This plots the oscillatory topomaps of a particular band
    eFrame.plot_band_stats(do_band='Alpha')
    
    
    #%%
    eFrame.band_distr() # this does the violin plots
    #eFrame.pca_decomp(direction='channels',condit='OnT',bl_correct=True,pca_type='rpca')
    
    eFrame.plot_pca_decomp(approach='rpca')
    #%%
    eFrame.train_binSVM()
    #eFrame.assess_binSVM()
    eFrame.analyse_binSVM(approach='rpca')
    #%%
    #eFrame.interval_stats(do_band='Alpha')
    eFrame.psd_stats(chann_list=[])

#%%
## Do some coherence measures here

if do_coherence:
    CSD_dict,PLV_dict = eFrame.coher_stat(pt_list=pt_list,chann_list=[])
    
    
    #%%
    #Package for pickle, this needs to be folded into the coher_stat method
    coh_measures = {'CSD':CSD_dict,'PLV':PLV_dict}
    with open('/tmp/DBS'+pt_list[0]+'_coh_dict.pickle','wb') as handle:
        pickle.dump(coh_measures,handle,protocol=pickle.HIGHEST_PROTOCOL)