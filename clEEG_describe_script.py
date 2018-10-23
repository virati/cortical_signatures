#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 16:32:20 2018

@author: virati
A REWRITE of the Cleaned EEG-Descriptive Pipeline (SCRIPT)
"""

#import sys
#sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/')
#import DBS_Osc as dbo

from proc_dEEG import proc_dEEG
from EEG_Viz import plot_3d_scalp

import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('paper')
sns.set(font_scale=4)
sns.set_style('white')

import pickle
import cmocean
#%%
pt_list = ['907']
#The feature vector, in this case the frequencies
fvect = np.linspace(0,500,513)

eFrame = proc_dEEG(pts=pt_list,procsteps='conservative',condits=['OnT','OffT'])
eFrame.extract_feats(polyorder=0)
#eFrame.gen_OSC_stack()

#%%
eFrame.band_stats(do_band='Alpha')

#%%
#eFrame.interval_stats(do_band='Alpha')
#eFrame.psd_stats(chann_list=[])

#%%
## Do some coherence measures here

CSD_dict,PLV_dict = eFrame.coher_stat(pt_list=pt_list,chann_list=[])


#%%
#Package for pickle
coh_measures = {'CSD':CSD_dict,'PLV':PLV_dict}
with open('/tmp/DBS'+pt_list[0]+'_coh_dict.pickle','wb') as handle:
    pickle.dump(coh_measures,handle,protocol=pickle.HIGHEST_PROTOCOL)