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
from DBS_Osc import nestdict

import pdb

sampling='DS500'

Targeting = defaultdict(dict)
Targeting['All'] = {
        
        '906':{
                'OnT':{
                        #'fname':'/home/virati/MDD_Data/hdEEG/Continuous/DS500/DBS906_TurnOn_Day1_Sess1_20150827_024013_tds.mat'
                        'fname':'/home/virati/MDD_Data/hdEEG/Continuous/Targeting/B04/DBS906/DBS906_TurnOn_Day1_Sess1_20150827_024013_OnTarget.mat'
                        },
                'OffT':{
                        'fname':'/home/virati/MDD_Data/hdEEG/Continuous/Targeting/B04/DBS906/DBS906_TurnOn_Day1_Sess2_20150827_041726_OffTarget.mat'
                        },
                
                'Volt':{
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
            for condit in do_condits:
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
                self.virtual_chann = nestdict()
                self.virtual_chann_loc = nestdict()
                
                del(container)
                
        self.do_pts = do_pts
        
        
    def re_ref(self,scheme='local',do_condits=['OnT','OffT']):
        print('Local Referencing...')
        
        #do a very simple lowpass filter at 1Hz
        hpf_cutoff=1/(self.fs/2)
        bc,ac = sig.butter(3,hpf_cutoff,btype='highpass',output='ba')
        
        for condit in do_condits:
            for pt in self.do_pts:
                if scheme == 'local':
                    dist_matr = neigh_mont.return_cap_L(dth=3)
                    
                    dataref = self.data_dict[pt][condit]
                    post_ref = neigh_mont.reref_data(dataref,dist_matr)
                
                    
                elif scheme == 'diff':
                    dist_matr = neigh_mont.return_cap_L(dth=3)
                    dataref = self.data_dict[pt][condit]
                    post_ref = neigh_mont.diff_reref(dataref,dist_matr)
                
                self.data_dict[pt][condit] = post_ref
                
                
                
                #pdb.set_trace()
                self.virtual_chann[pt][condit] = sig.filtfilt(bc,ac,np.array([vchann[0] for vchann in post_ref]))
                self.virtual_chann_loc[pt][condit] = np.array([vchann[1] for vchann in post_ref])
    
    def SG_Transform(self,nperseg=2**10,noverlap=2**10-10,ctype='real',do_condits=['OnT','OffT']):
        do_pts = self.do_pts
        if ctype == 'virtual':
            data_source = self.virtual_chann
        elif ctype == 'real':
            data_source = self.data_dictsud
        
        for pt in do_pts:
            for condit in do_condits:
                for cc in [32]:
                    
                    SGc = dbo.TF_Domain(data_source[pt][condit][cc,:],fs=self.fs,noverlap=noverlap,nperseg=nperseg)
                    plt.figure()
                    plt.subplot(211)
                    plt.pcolormesh(SGc['T'],SGc['F'],np.log10(SGc['SG']))
                    plt.subplot(212)
                    plt.plot(SGc['F'],np.log10(SGc['SG'][:,500]))
#%%
                    
                    
sEEG = streamEEG(fs=1000,ds_fact=2,do_pts=['906'])
sEEG.re_ref(scheme='diff')

#sEEG.re_ref()
#%%
sEEG.SG_Transform(nperseg=2**11,noverlap=2**11-50,ctype='virtual')

        

