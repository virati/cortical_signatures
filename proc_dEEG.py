#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 21:02:09 2018

@author: virati
Main Class for Processed dEEG Data
"""

import sys
sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/')
import DBS_Osc as dbo

from collections import defaultdict
import mne
from scipy.io import loadmat
import pdb
import numpy as np

import scipy.stats as stats
import matplotlib.pyplot as plt
plt.close('all')

from EEG_Viz import plot_3d_scalp

import seaborn as sns
sns.set()
sns.set_style("white")

from DBS_Osc import nestdict

from statsmodels import robust

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
        '908':{
                'OnT':'/home/virati/MDD_Data/hdEEG/Segmented/Targeting_B4/conservative/DBS908_TO_onTAR_bcr_LP_seg_mff_cln_bcr_ref.mat',
                'OffT':'/home/virati/MDD_Data/hdEEG/Segmented/Targeting_B4/conservative/DBS908_TO_offTAR_bcr_MU_seg_mff_cln_ref_1.mat'
                }
        }
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
        '908':{
                'OnT':'/home/virati/MDD_Data/hdEEG/Segmented/Targeting_B4/liberal/DBS908_TO_onTAR_bcr_LP_seg_mff_cln_ref.mat',
                'OffT':'/home/virati/MDD_Data/hdEEG/Segmented/Targeting_B4/liberal/DBS908_TO_offTAR_bcr_MU_seg_mff_cln_ref.mat'
                }
        }

keys_oi = {'OnT':["Off_3","BONT"],'OffT':["Off_3","BOFT"]}

class proc_dEEG:
    def __init__(self,pts,procsteps='liberal',condits=['OnT','OffT']):
        #load in the procsteps files
        ts_data = defaultdict(dict)
        
        for pt in pts:
            ts_data[pt] = defaultdict(dict)
            for condit in condits:
                ts_data[pt][condit] = defaultdict(dict)
                
                temp_data = loadmat(TargetingEXP[procsteps][pt][condit])
                
                for epoch in keys_oi[condit]:
                    ts_data[pt][condit][epoch] = temp_data[epoch]

        self.fs = temp_data['EEGSamplingRate'][0][0]
        self.donfft = 2**11
        self.fvect = np.linspace(0,self.fs/2,self.donfft/2+1)
        
        self.ts_data = ts_data
        self.pts = pts
        self.condits = condits
       
        self.ch_order_list = range(257)
        
        
        #sloppy containers for the outputs of our analyses        
        self.psd_trans = {pt:{condit:{epoch:[] for epoch in keys_oi} for condit in self.condits} for pt in self.pts}
        self.PSD_diff = {pt:{condit:[] for condit in self.condits} for pt in self.pts}
        self.PSD_var = {pt:{condit:[] for condit in self.condits} for pt in self.pts}
        
        self.Feat_trans = {pt:{condit:{epoch:[] for epoch in keys_oi} for condit in self.condits} for pt in self.pts}
        self.Feat_diff = {pt:{condit:[] for condit in self.condits} for pt in self.pts}
        self.Feat_var = {pt:{condit:[] for condit in self.condits} for pt in self.pts}
        
        self.eeg_locs = mne.channels.read_montage('/home/virati/Dropbox/GSN-HydroCel-257.sfp')
    
    def extract_feats(self,polyorder=4):
        pts = self.pts
        feat_dict = defaultdict(dict)
        for pt in pts:
            feat_dict[pt] = defaultdict(dict)
            for condit in self.condits:
                feat_dict[pt][condit] = defaultdict(dict)
                for epoch in keys_oi[condit]:
                    #find the mean for all segments
                    data_matr = self.ts_data[pt][condit][epoch] #should give us a big matrix with all the crap we care about
                    data_dict = {ch:data_matr[ch,:,:].squeeze() for ch in range(data_matr.shape[0])} #transpose is done to make it segxtime
                    
                    seg_psds,_ = dbo.gen_psd(data_dict,Fs=self.fs,nfft=self.donfft,polyord=polyorder)
                    
                    #gotta flatten the DICTIONARY, so have to do it carefully
                    
                    PSD_matr = np.array([seg_psds[ch] for ch in self.ch_order_list])
                    
                    #find the variance for all segments
                    feat_dict[pt][condit][epoch] = PSD_matr
        
        self.feat_dict = feat_dict
        
    def compute_diff(self,take_mean=True):
        avg_psd = nestdict()
        avg_change = nestdict()
        var_psd = nestdict()
        
        for pt in self.pts:
            #avg_psd[pt] = defaultdict(dict)
            #avg_change[pt] = defaultdict(dict)
            for condit in self.condits:
               #average all the epochs together
                avg_psd[pt][condit] = {epoch:np.median(self.feat_dict[pt][condit][epoch],axis=1) for epoch in self.feat_dict[pt][condit].keys()}
                #if you want variance
                #var_psd[pt][condit] = {epoch:np.var(self.feat_dict[pt][condit][epoch],axis=1) for epoch in self.feat_dict[pt][condit].keys()}
                #if you want Mean Absolute Deviance
                var_psd[pt][condit] = {epoch:robust.mad(self.feat_dict[pt][condit][epoch],axis=1) for epoch in self.feat_dict[pt][condit].keys()}
                keyoi = keys_oi[condit][1]
                
                avg_change[pt][condit] = 10*(np.log10(avg_psd[pt][condit][keyoi]) - np.log10(avg_psd[pt][condit]['Off_3']))
                
                
        self.feat_diff = avg_change
        self.feat_avg = avg_psd
        #This is really just a measure of how dynamic the underlying process is, not of particular interest for Aim 3.1, maybe 3.3
        self.feat_var = var_psd
        
    def NEWcompute_diff(self):
        avg_change = {pt:{condit:10*(np.log10(avg_psd[pt][condit][keys_oi[condit][1]]) - np.log10(avg_psd[pt][condit]['Off_3'])) for pt,condit in itertools.product(self.pts,self.condits)}}
   
    def pop_response(self):
        all_psds = nestdict()
        pop_lvl = nestdict()
        #pop_psds = defaultdict(dict)
        #pop_psds_var = defaultdict(dict)
        
        for condit in self.condits:
            all_psds[condit] = np.array([rr[condit] for pt,rr in self.feat_diff.items()])
            
            pop_lvl['Mean'][condit] = np.mean(all_psds[condit],axis=0)
            pop_lvl['Var'][condit] = np.var(all_psds[condit],axis=0)
        
        self.pop_stats = pop_lvl
    
    def do_pop_stats(self):
        self.reliablePSD = nestdict()
        for condit in self.condits:
            weighedPSD = np.divide(self.pop_stats['Mean'][condit].T,np.sqrt(self.pop_stats['Var'][condit].T))
            #do subtract weight
            subtrPSD = self.pop_stats['Mean'][condit].T - 2*np.sqrt(self.pop_stats['Var'][condit].T)
        
            #find where the channels are above threshold in a given band
            #go to each channel and find the mean change in a band
            
        
            self.reliablePSD[condit] = {'cPSD':subtrPSD,'CMask':chann_mask}
    def plot_pop_stats(self):
        for condit in self.condits:
            plt.figure()
            plt.subplot(311)
            plt.plot(self.fvect,self.pop_stats['Mean'][condit].T)
            plt.title('Mean')
            plt.subplot(312)
            plt.plot(self.fvect,np.sqrt(self.pop_stats['Var'][condit].T)/np.sqrt(3))
            plt.title('Standard Error of the Mean; n=3')
            
            
            plt.subplot(313)
            #do divide weight
            
            
            
            reliablePSD = self.reliablePSD[condit]['cPSD']
            
            plt.plot(self.fvect,reliablePSD)
            plt.title('SubtrPSD by Pop 6*std')
            
            plt.xlabel('Frequency (Hz)')
            #plt.subplot(313)
            #plt.hist(weighedPSD,bins=50)
            
            plt.suptitle(condit + ' population level')
    
    def plot_diff(self):
        
        for pt in self.pts:
            plt.figure()
            plt.subplot(221)
            plt.plot(self.fvect,self.feat_diff[pt]['OnT'].T)
            plt.title('OnT')
            
            plt.subplot(222)
            plt.plot(self.fvect,self.feat_diff[pt]['OffT'].T)
            plt.title('OffT')
            
            plt.subplot(223)
            plt.plot(self.fvect,10*np.log10(self.feat_var[pt]['OnT']['BONT'].T))
            
            
            plt.subplot(224)
            plt.plot(self.fvect,10*np.log10(self.feat_var[pt]['OffT']['BOFT'].T))
            
   
            plt.suptitle(pt)
    
    def GMM_train(self,condit='OnT'):
        #gnerate our big matrix of observations; Should be 256(chann)x4(feats)x(segxpatients)(observations)
        pass
    
    def DEPRextract_feats(self):
        donfft = self.donfft
        pts = self.pts
        for pt in pts:
            var_matr = defaultdict(dict)
            for condit in self.condits:
                med_matr = defaultdict(dict)
                var_matr = defaultdict(dict)
                
                med_feat_matr = defaultdict(dict)
                var_feat_matr = defaultdict(dict)
                
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
                        
                        ret_Fsegs = dbo.gen_psd(data_dict,Fs=1000,nfft=donfft)
                        
                        #need to ordered, build list
                        allchanns = [ret_Fsegs[ch] for ch in range(257)]
                        
                        med_matr[epoch] = 10*np.log10(np.squeeze(np.array(allchanns)))
                        self.psd_trans[pt][condit][epoch] = ret_Fsegs
                    #in methods 2, we do the PSD of each segment individually then we find the median across all segments
                    elif method == 2:
                        
                        chann_seglist = np.zeros((257,int(donfft/2+1),data_matr.shape[2]))
                        chann_segfeats = np.zeros((257,len(dbo.feat_order),data_matr.shape[2]))
                        for ss in range(data_matr.shape[2]):
                            data_dict = {ch:data_matr[ch,:,ss] for ch in range(data_matr.shape[0])}
                            ret_f = dbo.gen_psd(data_dict,Fs=1000,nfft=donfft)
                            #Push ret_f, the dictionary, into the matrix we need for further processing
                            for cc in range(257):
                                chann_seglist[cc,:,ss] = ret_f[cc]
                                #go to each channel and find the oscillatory band feature vector
                                chann_segfeats[cc,:,ss] = dbo.calc_feats(ret_f[cc],self.fvect)
                            
                        self.psd_trans[pt][condit][epoch] = chann_seglist
                        self.Feat_trans[pt][condit][epoch] = chann_segfeats
                        
                        med_matr[epoch] = np.median(10*np.log10(chann_seglist),axis=2)
                        var_matr[epoch] = np.var(10*np.log10(chann_seglist),axis=2)
                    
                        med_feat_matr[epoch] = np.median(10*np.log10(chann_segfeats),axis=2)
                        var_feat_matr[epoch] = np.var(10*np.log10(chann_segfeats),axis=2)
                
                assert med_matr[keys_oi[condit][1]].shape == (257,donfft/2+1)
                diff_matr = med_matr[keys_oi[condit][1]] - med_matr['Off_3']
                diff_feat = med_feat_matr[keys_oi[condit][1]] - med_feat_matr['Off_3']
                
                #Put the PSD information in the class structures
                self.PSD_var[pt][condit] = var_matr
                self.PSD_diff[pt][condit] = diff_matr
                #Put the Oscillator feature vector in the class structure
                self.Feat_diff[pt][condit] = diff_feat
                self.Feat_var[pt][condit] = var_feat_matr

        plot = False
        if plot:
            plt.figure()
            plt.subplot(211)
            plt.plot(self.fvect,med_matr['Off_3'].T)
            plt.subplot(212)
            plt.plot(self.fvect,med_matr[keys_oi[condit][1]].T)        
        
    def DEPRplot_diff(self,pt='906',condit='OnT',varweigh=False):
        diff_matr = self.PSD_diff[pt][condit]
        
        if varweigh:
            var_matr = self.PSD_var[pt][condit][keys_oi[condit][1]]
            plotdiff = (diff_matr / np.sqrt(var_matr)).T
            varweightext = ' WEIGHTED WITH 1/VAR'
        else:
            plotdiff = diff_matr.T
            varweightext = ''
            
        plt.figure()
        plt.plot(self.fvect,plotdiff,alpha=0.2)
        
        plt.xlim((0,150))
        plt.title('Difference from Pre-Stim to Stim')
        plt.suptitle(pt + ' ' + condit + varweightext)
    
    def plot_ontvsofft(self,pt='906'):
        if 'OffT' not in self.condits:
            raise ValueError
            
        condit_diff = (self.PSD_diff[pt]['OnT'] - self.PSD_diff[pt]['OffT']).T
        plt.figure()
        plt.plot(self.fvect,condit_diff,alpha=0.2)
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
        
    #This function quickly gets the power for all channels in each band
    def Full_feat_band(self):
        pass
        
        
    def plot_topo(self,vect,vmax=2,vmin=-2,label='',):
        plt.figure()
        mne.viz.plot_topomap(vect,pos=self.eeg_locs.pos[:,[0,1]],vmax=vmax,vmin=vmin,image_interp='none')
        plt.suptitle(label)