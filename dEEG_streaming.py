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
import matplotlib as mpl
plt.rcParams['image.cmap'] = 'jet'

import neigh_mont

import sys
sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/')
import DBS_Osc as dbo

import pdb

sampling='DS500'

Targeting = defaultdict(dict)
Targeting['All'] = {
        
        '906':{
                'OnT':{
                        #'fname':'/home/virati/MDD_Data/hdEEG/Continuous/DS500/DBS906_TurnOn_Day1_Sess1_20150827_024013_tds.mat'
                        'fname':'/home/virati/MDD_Data/hdEEG/Continuous/OnTOffT/B04/DBS906/DBS906_TurnOn_Day1_Sess1_20150827_024013.mat'
                        },
                'OffT':{
                        'fname':''
                        }
                },
        '908':{
                'OnT':{
                        'fname':'/home/virati/MDD_Data/hdEEG/Continuous/ALLMATS/DBS908_TurnOn_Day1_onTARGET_20160210_125231.mat'
                        },
                'OffT':{
                        'fname':'/home/virati/MDD_Data/hdEEG/Continuous/ALLMATS/DBS908_TurnOn_Day2_offTARGET_20160211_123540.mat'
                        }
                }
            }


class streamEEG:
        
    def __init__(self,do_pts=['906'],do_condits=['OnT','OffT'],ds_fact=1,fs=500):
        self.data_dict = {ev:{condit:[] for condit in do_condits} for ev in do_pts}
        
        self.fs = fs/ds_fact
        
        for pt in do_pts:
            for condit in ['OnT']:
                data_dict = defaultdict(dict)
                container = scio.loadmat(Targeting['All'][pt][condit]['fname'])
                dkey = [key for key in container.keys() if key[0:3] == 'DBS'][-1]
                                
                #data_dict = np.zeros((257,6*60*fs))
                #THIS IS FINE SINCE it's like a highpass with a DCish cutoff
                #10 * 60 * fs:18*60*fs
                tint = (np.array([238,1090]) * self.fs).astype(np.int)
                
                data_dict = sig.detrend(sig.decimate(container[dkey][:,tint[0]:tint[1]],ds_fact,zero_phase=True))
                #data_dict = data_dict - np.mean(data_dict,0)
                
                self.data_dict[pt][condit] = data_dict
                
                del(container)
                
        self.do_pts = do_pts
        
        
    def re_ref(self,scheme='local'):
        print('Local Referencing...')
        for pt in self.do_pts:
            if scheme == 'local':
                dist_matr = neigh_mont.return_cap_L(dth=3)
                
                dataref = self.data_dict[pt]['OnT']
                post_ref = neigh_mont.reref_data(dataref,dist_matr)
            
                
            elif scheme == 'diff':
                dist_matr = neigh_mont.return_cap_L(dth=3)
                dataref = self.data_dict[pt]['OnT']
                post_ref = neigh_mont.diff_reref(dataref,dist_matr)
            
            self.data_dict[pt]['OnT'] = post_ref
    
    def SG_Transform(self,nperseg=2**10,noverlap=2**10-10):
        do_pts = self.do_pts
        
        for pt in do_pts:
            for condit in ['OnT']:
                for cc in [32]:
                    
                    SGc = dbo.TF_Domain(self.data_dict[pt][condit][cc,:],fs=self.fs,noverlap=noverlap,nperseg=nperseg)
                    plt.figure()
                    plt.pcolormesh(SGc['T'],SGc['F'],np.log10(SGc['SG']))
#%%
                    
                    
sEEG = streamEEG(fs=1000,ds_fact=2,do_pts=['908'])
sEEG.re_ref(scheme='diff')

#sEEG.re_ref()

#sEEG.SG_Transform(nperseg=2**11,noverlap=2**11-50)

        

