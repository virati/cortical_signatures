#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 21:21:56 2018

@author: virati
This file loads in the preprocessed datafiles from AW preprocessing steps
Either the conservative versions or the non-conservative (liberal/all) versions
"""
import sys

sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/')
import DBS_Osc as dbo

from collections import defaultdict
import mne
from scipy.io import loadmat
import pdb
import numpy as np

import matplotlib.pyplot as plt
plt.close('all')
#%%

TargetingEXP = defaultdict(dict)
#TargetingEXP['conservative'] = {'905':0,'906':0,'907':0,'908':0}
TargetingEXP['conservative'] = {
        '905':{'OnT':'','OffT':''},
        '906':{
                'OnT':'/home/virati/MDD_Data/hdEEG/Segmented/Targeting_B4/conservative/DBS906_TO_onTAR_MU_HP_LP_seg_mff_cln_ref_1.mat',
                'OffT':'/home/virati/MDD_Data/hdEEG/Segmented/Targeting_B4/conservative/DBS906_TO_offTAR_bcr_LP_HP_seg_bcr_ref.mat'
                },
        '907':{
                'OnT':'/home/virati/MDD_Data/hdEEG/Segmented/Targeting_B4/conservative/DBS907_TO_onTAR_MU_seg_mff_cln_ref.mat',
                'OffT':'/home/virati/MDD_Data/hdEEG/Segmented/Targeting_B4/conservative/DBS907_TO_offTAR_MU_seg_mff_cln_ref.mat'
                },
        '908':0}
TargetingEXP['liberal'] = {
        '905':{'OnT':'','OffT':''},
        '906':{
                'OnT':'/home/virati/MDD_Data/hdEEG/Segmented/Targeting_B4/liberal/DBS906_TO_onTAR_MU_HP_LP_seg_mff_cln_ref.mat',
                'OffT':'/home/virati/MDD_Data/hdEEG/Segmented/Targeting_B4/liberal/DBS906_TO_offTAR_LP_seg_mff_cln_ref_1.mat'
                },
        '907':{
                'OnT':'/home/virati/MDD_Data/hdEEG/Segmented/Targeting_B4/liberal/DBS907_TO_onTAR_MU_seg_mff_cln_2ref.mat',
                'OffT':'/home/virati/MDD_Data/hdEEG/Segmented/Targeting_B4/liberal/DBS907_TO_offTAR_MU_seg_mff_cln_2ref.mat'
                },
        '908':0}

all_pts = ['906','907','908','905']

keys_oi = {'OnT':['Off_3','BONT'],'OffT':['Off_3','BOFT']}

class proc_dEEG:
    def __init__(self,procsteps='liberal',pts=all_pts,condits=['OnT','OffT']):
        #load in the procsteps files
        ts_data = defaultdict(dict)
        
        for pt in pts:
            ts_data[pt] = defaultdict(dict)
            for condit in condits:
                ts_data[pt][condit] = defaultdict(dict)
                try:
                    temp_data = loadmat(TargetingEXP[procsteps][pt][condit])
                except:
                    pdb.set_trace()
                for epoch in keys_oi[condit]:
                    ts_data[pt][condit][epoch] = temp_data[epoch]

        self.fs = temp_data['EEGSamplingRate']
        self.fvect = np.linspace(0,self.fs/2,2**9+1) 
                    
        self.ts_data = ts_data
        self.pts = pts
        self.condits = condits
        self.psd_trans = {pt:{condit:{epoch:[] for epoch in keys_oi} for condit in self.condits} for pt in self.pts}
        self.PSD_diff = {pt:{condit:[] for condit in self.condits} for pt in self.pts}
        self.PSD_var = {pt:{condit:[] for condit in self.condits} for pt in self.pts}
        
    def extract_feats(self):
        pts = self.pts
        for pt in pts:
            var_matr = defaultdict(dict)
            for condit in self.condits:
                med_matr = defaultdict(dict)
                var_matr[condit] = defaultdict(dict)
                #the size of this object should be
                for epoch in keys_oi[condit]:
                    data_matr = self.ts_data[pt][condit][epoch]
                    
                    #pdb.set_trace()
                    #plt.plot(data_dict[100].T)
                    #print(data_dict[100].shape)
                    # METHOD 1
                    method = 2
                    #Method one concatenates all segments then does the PSD
                    #this is not a good idea with the liberal preprocessing, since there are still artifacts
                    if method == 1:
                        #this next step gets rid of all the segments
                        data_dict = {ch:data_matr[ch,:,:].reshape(1,-1,order='F') for ch in range(data_matr.shape[0])}
                        
                        ret_Fsegs = dbo.gen_psd(data_dict,Fs=1000)
                        
                        #need to ordered, build list
                        allchanns = [ret_Fsegs[ch] for ch in range(257)]
                        
                        med_matr[epoch] = 10*np.log10(np.squeeze(np.array(allchanns)))
                        self.psd_trans[pt][condit][epoch] = ret_Fsegs
                    #in methods 2, we do the PSD of each segment individually then we find the median across all segments
                    elif method == 2:
                        
                        chann_seglist = np.zeros((257,513,data_matr.shape[2]))
                        for ss in range(data_matr.shape[2]):
                            data_dict = {ch:data_matr[ch,:,ss] for ch in range(data_matr.shape[0])}
                            ret_f = dbo.gen_psd(data_dict,Fs=1000)
                            for cc in range(257):
                                chann_seglist[cc,:,ss] = ret_f[cc]
                            
                        self.psd_trans[pt][condit][epoch] = chann_seglist
                        med_matr[epoch] = np.median(10*np.log10(chann_seglist),axis=2)
                        var_matr[epoch] = np.var(10*np.log10(chann_seglist),axis=2)
                    
                
                assert med_matr[keys_oi[condit][1]].shape == (257,513)
                diff_matr = med_matr[keys_oi[condit][1]] - med_matr['Off_3']
                
                self.PSD_var[pt][condit] = var_matr
                self.PSD_diff[pt][condit] = diff_matr

        plot = False
        if plot:
            plt.figure()
            plt.subplot(211)
            plt.plot(self.fvect,med_matr['Off_3'].T)
            plt.subplot(212)
            plt.plot(self.fvect,med_matr[keys_oi[condit][1]].T)        
        
    def plot_diff(self,pt='906',condit='OnT',varweigh=False):
        diff_matr = self.PSD_diff[pt][condit]
        
        
        if varweigh:
            var_matr = self.PSD_var[pt][condit][keys_oi[condit][1]]
            plotdiff = (diff_matr / np.sqrt(var_matr)).T
            varweightext = ' WEIGHTED WITH 1/VAR'
        else:
            plotdiff = diff_matr.T
            varweightext = ''
            
        plt.figure()
        plt.plot(self.fvect,plotdiff)
        plt.xlim((0,150))
        plt.title('Difference from Pre-Stim to Stim')
        plt.suptitle(pt + ' ' + condit + varweightext)
    
    def plot_ontvsofft(self,pt='906'):
        if 'OffT' not in self.condits:
            raise ValueError
            
        condit_diff = (self.PSD_diff[pt]['OnT'] - self.PSD_diff[pt]['OffT']).T
        plt.figure()
        plt.plot(self.fvect,condit_diff)
        plt.xlim((0,150))
        plt.title('Difference from OnT and OffT')
        plt.suptitle(pt)
    
    def plot_chann_var(self,pt='906',condit='OnT'):
        plt.figure()
        plt.subplot(121)
        plt.plot(self.PSD_var[pt][condit]['Off_3'].T)
        plt.xlim((0,150))
        plt.subplot(122)
        plt.plot(self.PSD_var[pt][condit][keys_oi[condit][1]].T)
        plt.xlim((0,150))
        
        plt.suptitle(pt + ' ' + condit)
                
                
#%%
        
#UNIT TEST
SegEEG = proc_dEEG(pts=['906'],procsteps='conservative',condits=['OffT'])
SegEEG.extract_feats()

#%%
for condit in ['OffT']:
    SegEEG.plot_diff(pt='906',varweigh=False,condit=condit)
    SegEEG.plot_chann_var(pt='906',condit=condit)
    #SegEEG.plot_ontvsofft(pt='906')
        


