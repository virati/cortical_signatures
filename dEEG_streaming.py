#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 21:24:09 2018

@author: virati
dEEG Continuous
Load in continuous, raw dEEG from the mat converted files
"""

from stream_dEEG import *

sEEG = streamEEG(ds_fact=2,pt='908',condit='OnT',spotcheck=True)
#sEEG.plot_TF(chann=32)
#%%

sEEG.seg_PSDs()

#%%
sEEG.calc_baseline()
#%%
sEEG.classify_segs()

#%%
#sEEG.re_ref(scheme='local')

#%%
#DO STREAMING, SEGMENTED Osc Band Calculations

#sEEG.re_ref()
#%%
#sEEG.SG_Transform(nperseg=2**11,noverlap=2**11-50,ctype='virtual')
#sEEG.SG_Transform(nperseg=2**11,noverlap=2**11-50,ctype='real')

        

