#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 21:24:09 2018

@author: virati
dEEG Continuous
Load in continuous, raw dEEG from the mat converted files
"""

import scipy.io as scio
import numpy as np
import pandas as pds
from collections import defaultdict
import scipy.signal as sig
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/')
import DBS_Osc as dbo

import pdb

sampling='DS500'

Targeting = defaultdict(dict)
Targeting['All'] = {
        '906':{
                'OnT':{
                        'fname':'/home/virati/MDD_Data/hdEEG/Continuous/DS500/DBS906_TurnOn_Day1_Sess3_20150827_051619_tds.mat'
                        },
                'OffT':{
                        'fname':''
                        }
                }
                }


class streamEEG:
        
    def __init__(self,do_pts=['906'],do_condits=['OnT','OffT'],ds_fact=1):
        self.data_dict = {ev:{condit:[] for condit in do_condits} for ev in do_pts}
        
        for pt in do_pts:
            for condit in ['OnT']:
                data_dict = defaultdict(dict)
                container = scio.loadmat(Targeting['All'][pt][condit]['fname'])
                dkey = [key for key in container.keys() if key[0:3] == 'DBS'][-1]
                
                
                data_dict = np.zeros((257,6*60*1000))
                data_dict = sig.detrend(sig.decimate(container[dkey][:,9 * 60 * 1000:15*60*1000],ds_fact))
                data_dict = data_dict - np.mean(data_dict,0)
                
                self.data_dict[pt][condit] = data_dict
                
                del(container)
                
        self.do_pts = do_pts
        self.fs = 500/ds_fact
    
    def SG_Transform(self):
        do_pts = self.do_pts
        
        for pt in do_pts:
            for condit in ['OnT']:
                for cc in [32]:
                    SGc = dbo.TF_Domain(self.data_dict[pt][condit][cc,:],fs=self.fs,noverlap=500)
                    plt.figure()
                    plt.pcolormesh(SGc['T'],SGc['F'],np.log10(SGc['SG']))
                
sEEG = streamEEG()

#%%
sEEG.SG_Transform()

        

