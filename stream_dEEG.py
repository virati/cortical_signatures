#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 17:41:48 2018

@author: virati
Streaming Class
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

import pickle

from sklearn import mixture
from sklearn.decomposition import PCA
from sklearn import svm


Targeting = defaultdict(dict)
Targeting['All'] = {
        
        '906':{
                'OnT':{
                        #'fname':'/home/virati/MDD_Data/hdEEG/Continuous/DS500/DBS906_TurnOn_Day1_Sess1_20150827_024013_tds.mat'
                        'fname':'/home/virati/MDD_Data/hdEEG/Continuous/Targeting/B04/DBS906/DBS906_TurnOn_Day1_Sess1_20150827_024013_OnTarget.mat',
                        'lfp':''
                        },
                'OffT':{
                        'fname':'/home/virati/MDD_Data/hdEEG/Continuous/Targeting/B04/DBS906/DBS906_TurnOn_Day1_Sess2_20150827_041726_OffTarget.mat',
                        'lfp':''
                        },
                
                'Volt':{
                        #
                        #'fname':'/home/virati/MDD_Data/hdEEG/Continuous/ALLMATS/DBS906_TurnOn_Day2_Sess3_Sess4_20150828_043231_VoltageAndFreq.mat'
                        'fname':'/home/virati/MDD_Data/hdEEG/Continuous/ALLMATS/DBS906_TurnOn_Day2_Sess2_20150828_032515_CurrentSweep.mat',
                        'lfp':''
                        }
                },
        '908':{
                'OnT':{
                        'fname':'/home/virati/MDD_Data/hdEEG/Continuous/Targeting/B04/DBS908/DBS908_TurnOn_Day1_onTARGET_20160210_125231.mat',
                        },
                'OffT':{
                        'fname':'/home/virati/MDD_Data/hdEEG/Continuous/Targeting/B04/DBS908/DBS908_TurnOn_Day2_offTARGET_20160211_123540.mat',
                        },
                'Volt':{'fname':'',
                        'lfp':''
                        }
                },
        '907':{
                'OnT':{
                        'fname':'/home/virati/MDD_Data/hdEEG/Continuous/ALLMATS/DBS907_TurnOn_Day1_onTARGET_20151216_105913.mat',
                        'lfp':''
                        },
                'OffT':{
                        'fname':'/home/virati/MDD_Data/hdEEG/Continuous/ALLMATS/DBS907_TurnOn_Day2_offTARGET_20151217_094245.mat',
                        'lfp':''
                        },
                'Volt':{
                        #'fname':'/home/virati/MDD_Data/hdEEG/Continuous/ALLMATS/DBS907_TurnOn_Day2_Voltage_20151217_102952.mat'
                        'fname':'/home/virati/MDD_Data/hdEEG/Continuous/ALLMATS/DBS907_TurnOn_Day3_Current_20151218_092443.mat',
                        'lfp':''
                        }
                },
        '901':{
            'OnT':{
                    'fname':'/tmp/DBS901-02 Eyes 20140520 1600.mat',
                    'lfp':''
                    },
            'OffT':{
                    'fname':'',
                    'lfp':''
                    },
            'Volt':{
                    'fname':'',
                    'lfp':''
                    }
            }
                
                
            }

class EEG_check:
    def __init__(self,pt='908',condit='OnT',ds_fact=1,fs=500,spotcheck=False):
        pass

class streamEEG:
        
    def __init__(self,pt='908',condit='OnT',ds_fact=1,spotcheck=False):
        #self.data_dict = {ev:{condit:[] for condit in do_condits} for ev in do_pts}
        
        self.donfft = 2**10
        
        
        data_dict = defaultdict(dict)
        container = scio.loadmat(Targeting['All'][pt][condit]['fname'])
        dkey = [key for key in container.keys() if key[0:3] == 'DBS'][-1]
        fs = container['EEGSamplingRate']
               
                   
        #data_dict = np.zeros((257,6*60*fs))
        #THIS IS FINE SINCE it's like a highpass with a DCish cutoff
        #10 * 60 * fs:18*60*fs
        snippet = True
        if snippet:
            #906
            if pt == '906' and condit == 'OnT':
                #tint = (np.array([238,1090]) * self.fs).astype(np.int)
                start_time = 4
            elif pt == '906' and condit =='OffT':
                start_time = 3
            elif pt == '908' and condit == 'OnT':
                #tint = (np.array([1000,1800]) * self.fs).astype(np.int)
                start_time = 16
            elif pt == '908' and condit == 'OffT':
                start_time = 0
            elif pt == '907':
                start_time = 0
            elif pt == '901':
                start_time = 0
            tlim = np.array((start_time,start_time + 16)) * 60 #in seconds
                        
                        
        tint = (tlim*fs).astype(np.int)[0]
        
        data_matr = sig.detrend(sig.decimate(container[dkey][:,tint[0]:tint[1]],ds_fact,zero_phase=True))
        #data_dict = data_dict - np.mean(data_dict,0)
        
        self.fs = container['EEGSamplingRate'][0] / ds_fact
        del(container)
                
        #self.data_dict[pt][condit] = data_dict
        self.data_matr = data_matr
        self.data_matr[256,:] = np.random.normal(size=self.data_matr[256,:].shape)
        self.re_ref()
        self.data_matr = self.re_ref_data
        
        #can add LFP and interpolate HERE?!
        
        
        
        #Proceed with standard code for EEG that should not break with LFP addition
        self.tvect = np.linspace(tlim[0],tlim[1],data_matr.shape[1])
        self.fvect = np.linspace(0,self.fs/2,self.donfft/2+1)
                

        self.re_ref_data = nestdict()
        self.pt = pt
        self.condit = condit
        
    def re_ref(self,scheme='local'):
        print('Local Referencing...')
        
        #do a very simple lowpass filter at 1Hz
        hpf_cutoff=1/(self.fs/2)
        bc,ac = sig.butter(3,hpf_cutoff,btype='highpass',output='ba')
        
        if scheme == 'local':
            dist_matr = neigh_mont.return_cap_L(dth=3)
            
            dataref = self.data_matr
            post_ref = neigh_mont.reref_data(dataref,dist_matr)        
            
        
        self.re_ref_data = post_ref
                
    def Osc_state(self):
        pass
    
    def make_segments(self):
        pass
        
    def seg_PSDs(self):
        tvect = self.tvect
        max_idx = tvect.shape[0]
        int_len = 6*int(self.fs)
        
        
        idxs = range(0,max_idx,int_len)
        idxs = idxs[:-1]
        num_segs = len(idxs)
        
        self.psd_matr = np.zeros((257,num_segs,int(self.donfft/2)+1))
        self.osc_matr = np.zeros((257,num_segs,len(dbo.feat_order)))
        self.stim_feat = np.zeros((num_segs))
        
        for ii,idx in enumerate(idxs):
            #print('Transforming segment ' + str(ii) + ' at ' + str(idx))
            seg_dict = {ch:self.data_matr[ch,idx:idx+int_len].squeeze().reshape(-1,1) for ch in range(257)}
            #generate the psd
            psd_vect = dbo.gen_psd(seg_dict,Fs=self.fs,nfft=self.donfft)
            self.psd_matr[:,ii,:] = np.array([psd_vect[ch] for ch in range(257)])
            #self.stim_feat[ii] = dbo.calc_feats(self.psd_matr[:,ii,:],self.fvect,dofeats=['Stim'])[0]
            
            #subtract out the polynom
            
            postpoly = dbo.poly_subtr(psd_vect,self.fvect)
            
            
            out_vect = dbo.calc_feats(postpoly,self.fvect,dofeats=['Delta','Theta','Alpha','Beta','Gamma1','Stim'])
            
            
            self.osc_matr[:,ii,:] = out_vect[0:5,:].T
            
            self.stim_feat[ii] = out_vect[-1,0]
            
            
        #seg_starts = tvect[0::2*self.fs]
    def load_classifier(self,ctype,train_type='cleaned'):
        if train_type == 'cleaned':
            self.clf = pickle.load(open('/home/virati/SVMModel_' + ctype,'rb'))
        elif train_type == 'stream':
            self.clf = pickle.load(open('/home/virati/Stream_SVMModel_' + ctype,'rb'))
    
    def calc_baseline(self):
        #go to every stim_feat segment WITHOUT stimulation and average them together. This is like a calibration
        
        no_stim_segs = self.stim_feat < 10
        stim_segs = np.logical_not( no_stim_segs)
        
        self.label_time = np.zeros((self.osc_matr.shape[1]))
        
        self.no_stim_median = np.median(self.osc_matr[:,no_stim_segs,:],axis=1)
        self.stim_matr = np.zeros_like(self.osc_matr)
        for ss in range(self.osc_matr.shape[1]):
            self.stim_matr[:,ss,:] = self.osc_matr[:,ss,:] - self.no_stim_median
            self.label_time[ss] = ss
        
        
        self.true_labels = np.zeros((self.osc_matr.shape[1]))
        
        if self.condit == 'OnT':
            label_val = 2
        elif self.condit == 'OffT':
            label_val = 1
        elif self.condit == 'Volt':
            label_val = 2
            
        self.label_val = label_val
        
        self.true_labels[stim_segs] = label_val
        
        
    def gen_test_matrix(self):
        test_matr = np.copy(self.stim_matr)
        test_matr = np.swapaxes(test_matr,0,1)
        test_matr = test_matr.reshape(-1,257*5,order='F')
        
        return test_matr
    
    def classify_segs(self,ctype='l2',train_type='cleaned'):
        self.load_classifier(ctype,train_type)
        
        test_matr = self.gen_test_matrix()
        
        
        self.pred_labels = self.clf.predict(test_matr)
        
        labmap = {'OnTON':2,'OffTON':1,'OFF':0}
        
        
        
        pred_nums = np.array([labmap[label] for label in self.pred_labels])
        #pred_nums = sig.medfilt(pred_nums.astype(np.float64),5)
        
        print('Accuracy: ' + str(sum(pred_nums == self.true_labels)/len(pred_nums)))
        #print(stats.mode(pred_nums))
        
        plt.figure()
        #plt.plot(self.pred_labels)
        plt.plot(pred_nums,label='Predicted',linewidth=3)
        plt.plot(self.true_labels,label='True',linewidth=5,alpha=0.6)
        plt.legend()
        
        
        #What percentage of the time where it's "ONT" is 
        loc_true = self.true_labels == self.label_val
        print('Prob Measure OnT | OnT Stim - True Positive')
        print(sum(pred_nums[loc_true] == self.label_val) / sum(loc_true))
        print('Prob Measure OnT | NOT OnT Stim - False Positive')
        print(sum(pred_nums[np.logical_not(loc_true)] == self.label_val) / sum(np.logical_not(loc_true)))
    
        #BAYES FLIP
        loc_pos = pred_nums == self.label_val
        print('Prob OnT | predicted On Target - PPV')
        print(sum(self.true_labels[loc_pos] == self.label_val)/sum(loc_pos))
        print('Prob OnT | predicted NOT OnTarget - NPV')
        print(sum(self.true_labels[np.logical_not(loc_pos)] == self.label_val)/sum(np.logical_not(loc_pos)))
    
        return (pred_nums,self.true_labels)
    
    def plot_TF(self,chann):
        in_x = self.data_matr[chann,:]
        
        nperseg = 2**10
        noverlap = 512
        
        freq,time,specg = sig.spectrogram(in_x,nperseg=nperseg,noverlap=noverlap,window=sig.get_window('blackmanharris',nperseg),fs=self.fs)
        
        
        #self.F = F
        #self.T = T
        #self.SG = SG
        
        
        plt.figure()
        plt.subplot(211)
        plt.pcolormesh(time,freq.squeeze(),10*np.log10(specg))
        plt.colorbar()
            
    