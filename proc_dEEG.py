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

import scipy.signal as sig

import scipy.stats as stats
import matplotlib.pyplot as plt
plt.close('all')

from EEG_Viz import plot_3d_scalp

import seaborn as sns

sns.set_context('paper')
sns.set(font_scale=4)
sns.set_style('white')


from DBS_Osc import nestdict

from statsmodels import robust

from sklearn import mixture
from sklearn.decomposition import PCA, FastICA
from sklearn import svm
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve

import pickle

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
        osc_dict = nestdict()
        
        for pt in pts:
            feat_dict[pt] = defaultdict(dict)
            
            for condit in self.condits:
                feat_dict[pt][condit] = defaultdict(dict)
                for epoch in keys_oi[condit]:
                    #find the mean for all segments
                    data_matr = self.ts_data[pt][condit][epoch] #should give us a big matrix with all the crap we care about
                    data_dict = {ch:data_matr[ch,:,:].squeeze() for ch in range(data_matr.shape[0])} #transpose is done to make it segxtime
                    
                    seg_psds = dbo.gen_psd(data_dict,Fs=self.fs,nfft=self.donfft,polyord=polyorder)
                    
                    #gotta flatten the DICTIONARY, so have to do it carefully
                    
                    PSD_matr = np.array([seg_psds[ch] for ch in self.ch_order_list])
                    
                    OSC_matr = np.zeros((seg_psds[0].shape[0],257,len(dbo.feat_order)))
                    #middle_osc = {chann:seg_psd for chann,seg_psd in seg_psds.items}
                    middle_osc = np.array([seg_psds[ch] for ch in range(257)])
                    
                    #have to go to each segment due to code
                    for ss in range(seg_psds[0].shape[0]):
                    
                        try:
                            state_return = dbo.calc_feats(middle_osc[:,ss,:],self.fvect).T
                            OSC_matr[ss,:,:] = np.array([state_return[ch] for ch in range(257)])
                        except Exception as e:
                            print(e)
                            pdb.set_trace()
                    
                    #find the variance for all segments
                    feat_dict[pt][condit][epoch] = PSD_matr
                    osc_dict[pt][condit][epoch] = OSC_matr
                    
                    #need to do OSCILLATIONS here
        
        #THIS IS THE PSDs RAW, not log transformed
        self.feat_dict = feat_dict
        self.osc_dict = osc_dict
    
    def train_simple(self):
        #Train our simple classifier that just finds the shortest distance
        self.signature = {'OnT':0,'OffT':0}
        self.signature['OnT'] = self.pop_osc_change['OnT'][dbo.feat_order.index('Alpha')] / np.linalg.norm(self.pop_osc_change['OnT'][dbo.feat_order.index('Alpha')])
        self.signature['OffT'] = self.pop_osc_change['OffT'][dbo.feat_order.index('Alpha')] / np.linalg.norm(self.pop_osc_change['OffT'][dbo.feat_order.index('Alpha')])
        
    
    def test_simple(self):
        #go to our GMM stack and, for each segment, determine the distance to the two conditions
        
        #Set up our signature
        cort_sig = {'OnT':self.Seg_Med[0]['OnT'],'OffT':self.Seg_Med[0]['OffT']}
        
        #Now let's generate a stacked set of the SAME 
        
        for condit in self.condits:
            seg_stack = self.GMM_Osc_stack[condit]
            seg_num = seg_stack.shape[1]
            OnT_sim[condit] = [None] * seg_num
            OffT_sim[condit] = [None] * seg_num
            
            for seg in range(seg_num):
                net_vect = seg_stack[:,seg,dbo.feat_order.index('Alpha')].reshape(-1,1)
                net_vect = net_vect/np.linalg.norm(net_vect)
                
                OnT_sim[condit][seg] = np.arccos(np.dot(net_vect.T,self.signature['OnT'].reshape(-1,1)) / np.linalg.norm(net_vect))
                OffT_sim[condit][seg] = np.arccos(np.dot(net_vect.T,self.signature['OffT'].reshape(-1,1)) / np.linalg.norm(net_vect))
                
            OnT_sim[condit] = np.array(OnT_sim[condit])
            OffT_sim[condit] = np.array(OffT_sim[condit])
            
        return (OnT_sim, OffT_sim)
    
    #this function will generate a big stack of all observations for a given condition across all patients
    
    def gen_OSC_stack(self,stack_type='all'):
        remap = {'Off_3':'OF','BONT':'ON','BOFT':'ON'}
        #big_stack = {key:0 for key in self.condits}
        big_stack = nestdict()
        for condit in self.condits:
            #want a stack for OnT and a stack for OffT
            big_stack[condit] = {remap[epoch]:{pt:self.osc_dict[pt][condit][epoch] for pt in self.pts} for epoch in keys_oi[condit]}
            
        if stack_type == 'all':
            pass
        
        self.big_stack_dict = big_stack
        return big_stack
    
    def simple_stats(self):
        ref_stack = self.big_stack_dict
        #Work with the Osc Dict data
        for condit in self.condits:
            ref_stack[condit]['Diff'] = defaultdict()#{key:[] for key in ([self.pts] + ['All'])}
            for epoch in ['OF','ON']:
                for pt in self.pts:
                    ref_stack[condit][epoch][pt + 'med'] = np.median(ref_stack[condit][epoch][pt],axis=0)
                    ref_stack[condit][epoch][pt + 'mad'] = robust.mad(ref_stack[condit][epoch][pt],axis=0)
                #stack all
                all_stack = [ref_stack[condit][epoch][pt] for pt in self.pts]
                
                ref_stack[condit][epoch]['MED'] = np.median(np.concatenate(all_stack,axis=0),axis=0)
                ref_stack[condit][epoch]['MAD'] = robust.mad(np.concatenate(all_stack,axis=0),axis=0)
            
            for pt in self.pts:
                ref_stack[condit]['Diff'][pt] = ref_stack[condit]['ON'][pt+'med'] - ref_stack[condit]['OF'][pt+'med']
            ref_stack[condit]['Diff']['All'] = ref_stack[condit]['ON']['MED'] - ref_stack[condit]['OF']['MED']
        
        all_stack = np.concatenate([np.concatenate([np.concatenate([ref_stack[condit][epoch][pt] for pt in self.pts],axis=0) for epoch in ['OF','ON']],axis=0) for condit in self.condits],axis=0)
        label_stack = [[[[condit+epoch for seg in ref_stack[condit][epoch][pt]] for pt in self.pts] for epoch in ['OF','ON']] for condit in self.condits]
        label_list = [item for sublist in label_stack for item in sublist]
        label_list = [item for sublist in label_list for item in sublist]
        label_list = [item for sublist in label_list for item in sublist]
        
        #go through each and if the last two characters are 'OF' -> 'OF' is label
        newlab = {'OffTOF':'OFF','OnTOF':'OFF','OnTON':'OnTON','OffTON':'OffTON'}
        label_list = [newlab[item] for item in label_list]
        
        self.SVM_stack = all_stack
        self.SVM_labels = np.array(label_list)
        
    
    def find_seg_covar(self):
        covar_matrix = nestdict()
        
        for condit in self.condits:
            seg_stack = sig.detrend(self.GMM_Osc_stack[condit],axis=1,type='constant')
            seg_num = seg_stack.shape[1]
            
            for bb,band in enumerate(dbo.feat_order):
                covar_matrix[condit][band] = []
                for seg in range(seg_num):
                    
                    net_vect = seg_stack[:,seg,bb].reshape(-1,1)
                    
                    cov_matr = np.dot(net_vect,net_vect.T)
                    covar_matrix[condit][band].append(cov_matr)
                    
                covar_matrix[condit][band] = np.array(covar_matrix[condit][band])
                    
        
        self.cov_segs = covar_matrix
                    
    def plot_seg_covar(self,band='Alpha'):
        for condit in self.condits:
            plt.figure()
            plt.subplot(211)
            plt.imshow(np.median(self.cov_segs[condit][band],axis=0),vmin=0,vmax=0.5)
            plt.colorbar()
            plt.title('Mean Covar')
            
            plt.subplot(212)
            plt.imshow(np.var(self.cov_segs[condit][band],axis=0),vmin=0,vmax=2)
            plt.colorbar()
            plt.title('Var Covar')
            
            plt.suptitle(condit)
    
    def OnT_v_OffT_MAD(self,band='Alpha'):
        Xdsgn = self.SVM_stack
        #THIS DOES MAD across the segments!!!!!!
        X_onT = Xdsgn[self.SVM_labels == 'OnTON',:,:].squeeze()
        X_offT = Xdsgn[self.SVM_labels == 'OffTON',:,:].squeeze()
        X_NONE = Xdsgn[self.SVM_labels == 'OFF',:,:].squeeze()
        
        OnT_MAD = robust.mad(X_onT,axis=0)
        OffT_MAD = robust.mad(X_offT,axis=0)
        NONE_MAD = robust.mad(X_NONE,axis=0)
        
        print('OnT segs: ' + str(X_onT.shape[0]))
        print('OffT segs: ' + str(X_offT.shape[0]))
        print('OFF segs: ' + str(X_NONE.shape[0]))
        
        self.Var_Meas = {'OnT':{'Med':np.median(X_onT,axis=0),'MAD':OnT_MAD},'OffT':{'Med':np.median(X_offT,axis=0),'MAD':OffT_MAD},'OFF':{'Med':np.median(X_NONE,axis=0),'MAD':NONE_MAD}}
        
    
    def pca_decomp(self,direction='channels',band='Alpha',condit='OnT',bl_correct=False):
        print('Doing PCA on the SVM Oscillatory Stack')
        #check to see if we have what variables we need
        Xdsgn = self.SVM_stack
        lbls = self.SVM_labels
        
        lblsdo = lbls == condit + 'ON'
        
        
        Xdo = Xdsgn[lblsdo,:,:]
        
        
        
        #what do we want to do with this now?
        #spatiotemporal PCA
        Xdo = np.median(Xdo,axis=0) 
        
        #BL_correct here is NOT patient specific bl correction
        if bl_correct:
            print('Correcting with baseline (OFF EEG)')
            #find the stim Off timepoints to subtract
            X_bl = np.median(Xdsgn[lbls=='OFF',:,:],axis=0)
            
            Xdo = Xdo - X_bl
            
            
        PCAdsgn = sig.detrend(Xdo,axis=0,type='constant')
        PCAdsgn = sig.detrend(PCAdsgn,axis=1,type='constant')
        
        pca = PCA()
        
        pca.fit(PCAdsgn)
        
        
        self.PCA_d = pca
        self.PCA_inX = Xdo
        
        PCA_X = pca.fit_transform(PCAdsgn)
        self.PCA_x = PCA_X
               
        
    
    def gen_GMM_priors(self,condit='OnT',mask_chann=False,band='Alpha'):
        #prior_change = self.pop_osc_mask[condit][dbo.feat_order.index(band)] * self.pop_osc_change[condit][dbo.feat_order.index(band)].reshape(-1,1)
        prior_change = self.pop_osc_change[condit][dbo.feat_order.index(band)].reshape(-1,1)
        if mask_chann:
            mask = self.median_mask
            prior_covar=np.dot(prior_change[mask],prior_change[mask].T) 
        else:
            prior_covar = np.dot(prior_change,prior_change.T)
        
        self.prior_covar = prior_covar
        return prior_covar
        
    
    def gen_GMM_Osc(self,GMM_stack):
        fvect = self.fvect
        
        feat_out = nestdict()
        
        for condit in self.condits:
            num_segs = GMM_stack[condit].shape[1]
            feat_out[condit] = np.zeros((257,num_segs,len(dbo.feat_order)))
            
            for ss in range(num_segs):
                
                feat_out[condit][:,ss,:] = dbo.calc_feats(10**(GMM_stack[condit][:,ss,:]/10).squeeze(),fvect).T
                
                
        #THIS CURRENTLY HAS nans
        GMM_Osc_stack = feat_out
        
        return {'Stack':GMM_Osc_stack}
        
    def shape_GMM_dsgn(self,inStack_dict,band='Alpha',mask_channs=False):
        segs_feats = nestdict()
        
        inStack = inStack_dict['Stack']
        
        
        assert inStack['OnT'].shape[2] < 10        
        for condit in self.condits:
            num_segs = inStack[condit].shape[1]
            
            #I think this makes it nsegs x nchann x nfeats?
            segs_chann_feats = np.swapaxes(inStack[condit],1,0)
            
            #this stacks all the ALPHAS for all channels together
            
            
            if mask_channs:
                chann_mask = self.median_mask
            else:
                chann_mask = np.array([True] * 257)
            
            #CHOOSE ALPHA HERE
            
            if band == 'Alpha':
                band_idx = dbo.feat_order.index(band)
                segs_feats[condit] = segs_chann_feats[:,chann_mask,band_idx]
                
            elif band == 'All':
                segs_feats[condit] = segs_chann_feats[:,chann_mask,:]
                
            #segs_feats = np.reshape(segs_chann_feats,(num_segs,-1),order='F')
            
            #We want a 257 dimensional vector with num_segs observations
        
        return segs_feats

    def pop_meds(self):
        print('Doing Population Meds/Mads on Oscillatory RESPONSES w/ PCA')
        dsgn_X = self.shape_GMM_dsgn(self.gen_GMM_Osc(self.gen_GMM_stack(stack_bl='normalize')['Stack']),band='All')
        
        X_med = nestdict()
        X_mad = nestdict()
        X_segnum = nestdict()
        #do some small simple crap here
        
        #Here we're averaging across axis zero which corresponds to 'averaging' across SEGMENTS
        for condit in self.condits:
            
            X_med[condit]= np.median(dsgn_X[condit],axis=0)
            #X_med[condit]= np.mean(dsgn_X[condit],axis=0)
            X_mad[condit] = robust.mad(dsgn_X[condit],axis=0)
            #X_mad[condit] = np.var(dsgn_X[condit],axis=0)
            X_segnum[condit] = dsgn_X[condit].shape[0]
        
        self.Seg_Med = (X_med,X_mad,X_segnum)
        
        weigh_mad = 0.3
        self.median_mask = (np.abs(self.Seg_Med[0]['OnT'][:,2]) - weigh_mad*self.Seg_Med[1]['OnT'][:,2] >= 0)
        
        #Do a quick zscore to zero out the problem channels
        chann_patt_zs = stats.zscore(X_med['OnT'],axis=0)
        outlier_channs = np.where(chann_patt_zs > 3)

    def do_ICA_fullstack(self):
        rem_channs = False     
        print('ICA Time')
        ICA_inX = X_med['OnT']
        if rem_channs:
            ICA_inX[outlier_channs,:] = np.zeros_like(ICA_inX[outlier_channs,:])
        
        PCA_inX = np.copy(ICA_inX)
        
        #Go ahead and do PCA here since the variables are already here
        
        #PCA SECTION
        #pca = PCA()

        
        #ICA
        ica = FastICA(n_components=5)
        ica.fit(ICA_inX)
        self.ICA_d = ica
        self.ICA_inX = ICA_inX
        self.ICA_x = ica.fit_transform(ICA_inX)
        
        
    def gen_GMM_stack(self,stack_bl=''):
        state_stack = nestdict()
        state_labels = nestdict()
        
        
        for condit in self.condits:
            state_stack[condit] = []
            keyoi = keys_oi[condit][1]
            
            if stack_bl == 'add':
                raise ValueError
                pt_stacks = [val[condit][keyoi] for pt,val in self.feat_dict.items()] 
                base_len = len(pt_stacks)
                pt_labels = [keyoi] * base_len
                
                pt_stacks += [val[condit]['Off_3'] for pt,val in self.feat_dict.items()]
                pt_labels += ['Off_3'] * (len(self.feat_dict.keys()) - base_len)
                #pt_labels = [keyoi for pt,val in self.feat_dict.items()] + ['Off_3' for pt,val in self.feat_dict.items()]
                self.num_gmm_comps = 3
            elif stack_bl == 'normalize':
                #bl_stack = [val[condit]['Off_3'] for pt,val in self.feat_dict.items()]
                #bl_stack = np.concatenate(bl_stack,axis=1)
                bl_stack = {pt:val[condit]['Off_3'] for pt,val in self.feat_dict.items()}
                
                pt_stacks = [val[condit][keyoi] - np.repeat(np.median(bl_stack[pt],axis=1).reshape(257,1,1025),val[condit][keyoi].shape[1],axis=1) for pt,val in self.feat_dict.items()]
                pt_labels = [[keyoi] * gork.shape[1] for gork in pt_stacks]
                self.num_gmm_comps = 2
            else:
                pt_stacks = [val[condit][keyoi] for pt,val in self.feat_dict.items()]
                pt_labels = [keyoi] * len(pt_stacks)
                self.num_gmm_comps = 2
                
            state_stack[condit] = np.concatenate(pt_stacks,axis=1)
            state_labels[condit] = pt_labels
            
        
        GMM_stack = state_stack
        GMM_stack_labels = state_labels
        self.GMM_stack_labels = state_labels
        return {'Stack':GMM_stack,'Labels':GMM_stack_labels}
    
    
    def do_response_PCA(self):
        pca = PCA()
        
        print("Using GMM Stack, which is baseline normalized within each patient")
        
        #This has both OnTarget and OffTarget conditions in the stack
        
        GMMStack = self.gen_GMM_stack(stack_bl='normalize')
        GMMOsc = self.gen_GMM_Osc(GMMStack['Stack'])
        
        
        
        PCA_inX = np.median(np.swapaxes(GMMOsc['Stack']['OnT'],0,1),axis=0)
        
        pca.fit(PCA_inX)
        self.PCA_d = pca
        self.PCA_inX = PCA_inX
        self.PCA_x = pca.fit_transform(PCA_inX)
        
    
    def plot_PCA_stuff(self):
        plt.figure();
        plt.subplot(221)
        plt.imshow(self.PCA_d.components_,cmap=plt.cm.jet,vmax=1,vmin=-1)
        plt.colorbar()
        
        plt.subplot(222)
        plt.plot(self.PCA_d.components_)
        plt.ylim((-1,1))
        plt.legend(['PC0','PC1','PC2','PC3','PC4'])
        plt.xticks(np.arange(0,5),['Delta','Theta','Alpha','Beta','Gamma1'])
        
        plt.subplot(223)
        plt.plot(self.PCA_d.explained_variance_ratio_)
        plt.ylim((0,1))
        
        for cc in range(2):
            fig=plt.figure()
            plot_3d_scalp(self.PCA_x[:,cc],fig,animate=False,unwrap=True)
            plt.title('Plotting component ' + str(cc))
            plt.suptitle('PCA rotated results for OnT')

    
    def plot_ICA_stuff(self):
        plt.figure()
        plt.subplot(221)
        plt.imshow(self.ICA_d.components_[:,:-1],cmap=plt.cm.jet,vmax=1,vmin=-1)
        plt.colorbar()
        plt.subplot(222)
        plt.plot(self.ICA_d.components_[:,:-1])
        plt.legend(['IC0','IC1','IC2','IC3','IC4'])
        plt.xticks(np.arange(0,5),['Delta','Theta','Alpha','Beta','Gamma1'])
        plt.subplot(223)
        
        plt.plot(self.ICA_d.mixing_)
        
        for cc in range(2):
            fig=plt.figure()
            plot_3d_scalp(self.ICA_x[:,cc],fig,animate=False,unwrap=True)
            plt.title('Plotting component ' + str(cc))
            plt.suptitle('ICA rotated results for OnT')
            
    def plot_meds(self,band='Alpha',flatten=True):
        print('Doing Population Level Medians and MADs')
        band_idx = dbo.feat_order.index(band)
        
        plt.figure()
        plt.plot(self.Seg_Med[0]['OnT'][:,band_idx],label='OnT')
        plt.fill_between(np.arange(257),self.Seg_Med[0]['OnT'][:,band_idx] - self.Seg_Med[1]['OnT'][:,band_idx]/np.sqrt(self.Seg_Med[2]['OnT']),self.Seg_Med[0]['OnT'][:,band_idx] + self.Seg_Med[1]['OnT'][:,band_idx]/np.sqrt(self.Seg_Med[2]['OnT']),alpha=0.4)
        print(self.Seg_Med[2]['OnT'])
        
        plt.plot(self.Seg_Med[0]['OffT'][:,band_idx],label='OffT')
        plt.fill_between(np.arange(257),self.Seg_Med[0]['OffT'][:,band_idx] - self.Seg_Med[1]['OffT'][:,band_idx]/np.sqrt(self.Seg_Med[2]['OffT']),self.Seg_Med[0]['OffT'][:,band_idx] + self.Seg_Med[1]['OffT'][:,band_idx]/np.sqrt(self.Seg_Med[2]['OffT']),alpha=0.4)
        print(self.Seg_Med[2]['OffT'])
        plt.title('Medians across Channels')
        plt.legend()
        
        
        for condit in self.condits:
            fig = plt.figure()
            #This is MEDS
            plot_3d_scalp(self.Seg_Med[0][condit][:,band_idx],fig,label=condit + '_med',animate=False,clims=(-0.2,0.2),unwrap=flatten)
            plt.suptitle('Median of Cortical Response across all ' + condit + ' segments | Band is ' + band)
        
        plt.figure()
        plt.plot(self.Seg_Med[1]['OnT'][:,band_idx],label='OnT')
        
        plt.plot(self.Seg_Med[1]['OffT'][:,band_idx],label='OffT')
        plt.title('MADs across Channels')
        plt.legend()
        for condit in self.condits:
            fig = plt.figure()
            #this is MADs
            plot_3d_scalp(self.Seg_Med[1][condit][:,band_idx],fig,label=condit + '_mad',animate=False,unwrap=flatten,clims=(0,1.0))
            plt.suptitle('MADs of Cortical Response across all ' + condit + ' segments | Band is ' + band)
        
        #Finally, for qualitative, let's look at the most consistent changes
        for condit in self.condits:
            weigh_mad = 0.4
            fig = plt.figure()
            masked_median = self.Seg_Med[0][condit][:,band_idx] * (np.abs(self.Seg_Med[0][condit][:,band_idx]) - weigh_mad*self.Seg_Med[1][condit][:,band_idx] >= 0).astype(np.int)
            plot_3d_scalp(masked_median,fig,label=condit + '_masked_median',animate=False,clims=(-0.1,0.1))
            plt.suptitle('Medians with small variances (' + str(weigh_mad) + ') ' + condit + ' segments | Band is ' + band)
        
        ## Figure out which channels have overlap
        ont_olap = np.array((self.Seg_Med[0]['OnT'][:,band_idx],self.Seg_Med[0]['OnT'][:,band_idx] - self.Seg_Med[1]['OnT'][:,band_idx]/np.sqrt(self.Seg_Med[2]['OnT']),self.Seg_Med[0]['OnT'][:,band_idx] + self.Seg_Med[1]['OnT'][:,band_idx]/np.sqrt(self.Seg_Med[2]['OnT'])))
        offt_olap = np.array((self.Seg_Med[0]['OffT'][:,band_idx],self.Seg_Med[0]['OffT'][:,band_idx] - self.Seg_Med[1]['OffT'][:,band_idx]/np.sqrt(self.Seg_Med[2]['OffT']),self.Seg_Med[0]['OffT'][:,band_idx] + self.Seg_Med[1]['OffT'][:,band_idx]/np.sqrt(self.Seg_Med[2]['OffT'])))
        
        for cc in range(257):
            np.hstack((ont_olap[1][cc],ont_olap[2][cc]))
        
        #do a sweep through to find the channels that don't overlap
        sweep_range = np.linspace(-0.3,0.3,100)
        
        ont_over = np.zeros_like(sweep_range)
        offt_over = np.zeros_like(sweep_range)
        
        for ss,sr in enumerate(sweep_range):
            ont_over[ss] = sum(self.Seg_Med[0]['OnT'][:,band_idx] > sr)
            offt_over[ss] = sum(self.Seg_Med[0]['OffT'][:,band_idx] > sr)
        
        plt.figure()
        plt.plot(sweep_range,ont_over)
        plt.plot(sweep_range,offt_over)
        
    def train_GMM(self):
        #shape our dsgn matrix properly
        intermed_X = self.gen_GMM_Osc(self.gen_GMM_stack(stack_bl='normalize')['Stack'])
        dsgn_X = self.shape_GMM_dsgn(intermed_X,mask_channs=True)
        

        #let's stack the two together and expect 3 components?
        #this is right shape: (n_samples x n_features)
        
        #break out our dsgn matrix
        condit_dict = [val for key,val in dsgn_X.items()]
        
        full_X = np.concatenate(condit_dict,axis=0)
        
        #setup our covariance prior from OUR OTHER ANALYSES
        
        covariance_prior = self.gen_GMM_priors(mask_chann=True)
        
        #when below reg_covar was 1e-1 I got SOMETHING to work
        #gMod = mixture.BayesianGaussianMixture(n_components=self.num_gmm_comps,mean_prior=self.Seg_Med[0]['OnT'],mean_precision_prior=0.1,covariance_type='full',covariance_prior=np.dot(self.Seg_Med[1]['OnT'].reshape(-1,1),self.Seg_Med[1]['OnT'].reshape(-1,1).T),reg_covar=1,tol=1e-2)#,covariance_prior=covariance_prior_alpha)
        
        #BAYESIAN GMM version
        gMod = mixture.BayesianGaussianMixture(n_components=self.num_gmm_comps,mean_precision_prior=0.1,covariance_type='full',reg_covar=1e-6,tol=1e-2)#,covariance_prior=covariance_prior_alpha)
        condit_mean_priors = [np.median(rr) for rr in condit_dict]
        
        gMod.means_ = condit_mean_priors
        gMod.covariance_prior_ = covariance_prior
        
        #STANDARD GMM        
        #gMod = mixture.GaussianMixture(n_components=self.num_gmm_comps)
        
        
        try:
            gMod.fit(full_X)
        except Exception as e:
            print(e)
            pdb.set_trace()
            
        
        self.GMM = gMod
        
        self.predictions = gMod.predict(full_X)
        self.posteriors = gMod.predict_proba(full_X)
    
    def train_newGMM(self,mask=False):
        num_segs = self.SVM_stack.shape[0]
        
        #generate a mask
        if mask:
            sub_X = self.SVM_stack[:,self.median_mask,:]
            dsgn_X = sub_X.reshape(num_segs,-1,order='C')
        else:
            dsgn_X = self.SVM_stack.reshape(num_segs,-1,order='C')
        
        
        #doing a one class SVM
        #clf = svm.OneClassSVM(nu=0.1,kernel="rbf", gamma=0.1)
        #clf = svm.LinearSVC(penalty='l2',dual=False)
        #gMod = mixture.BayesianGaussianMixture(n_components=3,mean_precision_prior=0.1,covariance_type='full',reg_covar=1e-6,tol=1e-2)#,covariance_prior=covariance_prior_alpha)
        gMod = mixture.BayesianGaussianMixture(n_components=3)
        
        #split out into test and train
        Xtr,Xte,Ytr,Yte = sklearn.model_selection.train_test_split(dsgn_X,self.SVM_labels,test_size=0.33)
        

        
        gMod.fit(Xtr,Ytr)
        
        #predict IN training set
        predlabels = gMod.predict(Xte)
        
        plt.figure()
        plt.plot(Yte,label='test')
        plt.plot(predlabels,label='predict')
        
        print(np.sum(np.array(Yte) == np.array(predlabels))/len(Yte))
        
    def train_SVM(self,mask=False):
        num_segs = self.SVM_stack.shape[0]
        
        #generate a mask
        if mask:
            #what mask do we want?
            #self.SVM_Mask = self.median_mask
            self.SVM_Mask = np.zeros((257,)).astype(bool)
            self.SVM_Mask[[238,237]] = True
            
            sub_X = self.SVM_stack[:,self.SVM_Mask,:]
            dsgn_X = sub_X.reshape(num_segs,-1,order='C')
        else:
            dsgn_X = self.SVM_stack.reshape(num_segs,-1,order='C')
        
        #Learning curve
        print('Learning Curve')
        tsize,tscore,vscore = learning_curve(svm.LinearSVC(penalty='l2',dual=False),dsgn_X,self.SVM_labels,train_sizes=np.linspace(0.2,1.0,5),cv=5)
        plt.figure()
        plt.plot(tsize,np.mean(tscore,axis=1))
        plt.plot(tsize,np.mean(vscore,axis=1))



        #doing a one class SVM
        #clf = svm.OneClassSVM(nu=0.1,kernel="rbf", gamma=0.1)
        clf = svm.LinearSVC(penalty='l2',dual=False)
        
        #split out into test and train
        Xtr,Xte,Ytr,Yte = sklearn.model_selection.train_test_split(dsgn_X,self.SVM_labels,test_size=0.33)
        

        
        clf.fit(Xtr,Ytr)
        
        #predict IN training set
        predlabels = clf.predict(Xte)
        
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(Yte,label='test')
        plt.plot(predlabels,label='predict')
        simple_accuracy = np.sum(np.array(Yte) == np.array(predlabels))/len(Yte)
        plt.title('SVM Results with Mask:' + str(mask) + ' ; Accuracy: ' + str(simple_accuracy))
        plt.legend()
        
        pickle.dump(clf,open('/tmp/SVMModel_l2','wb'))
        
        plt.subplot(2,1,2)
        conf_matrix = confusion_matrix(predlabels,Yte)
        plt.imshow(conf_matrix)
        plt.yticks(np.arange(0,3),['OFF','OffT','OnT'])
        plt.xticks(np.arange(0,3),['OFF','OffT','OnT'])
        plt.colorbar()
        
        
        self.SVM = clf
        self.SVM_dsgn_X = dsgn_X
        self.SVM_test_labels = predlabels
    
    def train_binSVM(self,mask=False):
        num_segs = self.SVM_stack.shape[0]
        print('DOING BINARY')
        #generate a mask
        if mask:
            #what mask do we want?
            #self.SVM_Mask = self.median_mask
            self.SVM_Mask = np.zeros((257,)).astype(bool)
            self.SVM_Mask[np.arange(216,239)] = True
            
            sub_X = self.SVM_stack[:,self.SVM_Mask,:]
            dsgn_X = sub_X.reshape(num_segs,-1,order='C')
        else:
            dsgn_X = self.SVM_stack.reshape(num_segs,-1,order='C')
        
        #doing a one class SVM
        #clf = svm.OneClassSVM(nu=0.1,kernel="rbf", gamma=0.1)
        
        

        
        
        
        
        #get rid of ALL OFF, and only do two labels
        OFFs = np.where(self.SVM_labels == 'OFF')
        dsgn_X = np.delete(dsgn_X,OFFs,0)
        SVM_labels = np.delete(self.SVM_labels,OFFs,0)
        
        #learning curve
        print(dsgn_X.shape)
        
        
        
        
        
        #split out into test and train
        Xtr,Xte,Ytr,Yte = sklearn.model_selection.train_test_split(dsgn_X,SVM_labels,test_size=0.90,random_state=0)
        
        tsize,tscore,vscore = learning_curve(svm.LinearSVC(penalty='l2',dual=False,C=1),Xtr,Ytr,train_sizes=np.linspace(0.4,1,10),shuffle=True,cv=5,random_state=0)
        plt.figure()
        plt.plot(tsize,np.mean(tscore,axis=1))
        plt.plot(tsize,np.mean(vscore,axis=1))
        
        
        #classifier time itself
        clf = svm.LinearSVC(penalty='l2',dual=False)
        #Fit the actual algorithm
        clf.fit(Xtr,Ytr)
        
        #predict IN training set
        predlabels = clf.predict(Xte)
        
        plt.figure()
        plt.subplot(2,1,1)
        
        Yten = np.copy(Yte)
        predlabelsn = np.copy(predlabels)
        
        #Fix labels for PLOTTING
        Yten[Yte == 'OffTON'] = 0
        Yten[Yte == 'OnTON'] = 1
        predlabelsn[predlabels == 'OffTON'] = 0
        predlabelsn[predlabels == 'OnTON'] = 1
        
        
        plt.imshow(np.vstack((Yten.astype(np.int),predlabelsn.astype(np.int))),aspect='auto')
        #plt.plot(Yte,label='test')
        #plt.plot(predlabels,label='predict')
        #simple_accuracy = np.sum(np.array(Yte) == np.array(predlabels))/len(Yte)
        score = clf.score(Xte,Yte)
        plt.title('SVM Results with Mask:' + str(mask) + ' ; Accuracy: ' + str(score))
        plt.legend()
        
        pickle.dump(clf,open('/tmp/SVMModel_l2','wb'))
        
        plt.subplot(2,1,2)
        conf_matrix = confusion_matrix(predlabels,Yte)
        plt.imshow(conf_matrix)
        plt.yticks(np.arange(0,2),['OffT','OnT'])
        plt.xticks(np.arange(0,2),['OffT','OnT'])
        plt.colorbar()
        
        
        self.binSVM = clf
        self.binSVM_dsgn_X = dsgn_X
        self.binSVM_test_labels = predlabels
        
        
    def compute_diff(self,take_mean=True):
        assert len(self.condits) >= 2
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
                
                
        self.psd_change = avg_change
        self.psd_avg = avg_psd
        #This is really just a measure of how dynamic the underlying process is, not of particular interest for Aim 3.1, maybe 3.3
        self.psd_var = var_psd
        
    def NEWcompute_diff(self):
        avg_change = {pt:{condit:10*(np.log10(avg_psd[pt][condit][keys_oi[condit][1]]) - np.log10(avg_psd[pt][condit]['Off_3'])) for pt,condit in itertools.product(self.pts,self.condits)}}
   
    #This goes to the psd change average and computed average PSD across all available patients
    def pop_response(self):
        psd_change_matrix = nestdict()
        population_change = nestdict()
        #pop_psds = defaultdict(dict)
        #pop_psds_var = defaultdict(dict)
        
        #Generate the full PSDs matrix and then find the MEAN along the axes for the CHANGE
        for condit in self.condits:
            #Get us a matrix of the PSD Changes
            psd_change_matrix[condit] = np.array([rr[condit] for pt,rr in self.psd_change.items()])
            
            population_change[condit]['Mean'] = np.mean(psd_change_matrix[condit],axis=0)
            population_change[condit]['Var'] = np.var(psd_change_matrix[condit],axis=0)
            
            
        self.pop_psd_change = population_change
    
    def do_pop_stats(self,band='Alpha',threshold=0.4):
        #self.reliablePSD = nestdict()
        avgOsc = nestdict()
        varOsc = nestdict()
        varOsc_mask = nestdict()
        
        for condit in self.condits:
            
            #First, let's average across patients
            
            #band = np.where(np.logical_and(self.fvect > 14, self.fvect < 30))
            
            avgOsc[condit] = dbo.calc_feats(10**(self.pop_psd_change[condit]['Mean']/10),self.fvect)
            varOsc[condit] = dbo.calc_feats(10**(self.pop_psd_change[condit]['Var']),self.fvect)
            varOsc_mask[condit] = np.array(np.sqrt(varOsc[condit]) < threshold).astype(np.int)
            
        self.pop_osc_change = avgOsc
        self.pop_osc_var = varOsc
        self.pop_osc_mask = varOsc_mask

    def do_pop_mask(self,threshold):
        cMask = nestdict()
        
        #weighedPSD = np.divide(self.pop_stats['Mean'][condit].T,np.sqrt(self.pop_stats['Var'][condit].T))
        #do subtract weight
        #THIS SHOULD ONLY BE USED TO DETERMINE THE MOST USEFUL CHANNELS
        #THIS IS A SHITTY THING TO DO
        for condit in self.condits:
            pre_mask = np.abs(self.pop_change['Mean'][condit].T) - threshold*np.sqrt(self.pop_change['Var'][condit].T)
            #which channels survive?
            cMask[condit] = np.array(pre_mask > 0).astype(np.int)
            
        cMask['Threshold'] = threshold
        
        self.reliability_mask = cMask
        
            
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
            
    
    def topo_wrap(self,band='Alpha',label='',condit='OnT',mask=False,animate=False):
        mainfig = plt.figure()
        
        if not mask:
            plot_3d_scalp(self.pop_osc_change[condit][dbo.feat_order.index(band)],mainfig,animate=animate,label=label,clims=(-1,1))
        else:
            try:
                plot_3d_scalp(self.pop_osc_mask[condit][dbo.feat_order.index(band)] * self.pop_osc_change[condit][dbo.feat_order.index(band)],mainfig,clims=(-1,1),animate=animate,label=label)
            except:
                pdb.set_trace()
        
            
        plt.suptitle(label)
        
        
    def topo_3d_chann_mask(self,band='Alpha',animate=True,pt='all',var_fixed=False,label=''):
        for condit in ['OnT','OffT']:
            #plot_vect = np.zeros((257))
            #plot_vect[self.reliablePSD[condit]['CMask']] = 1
            #plot_vect = self.reliablePSD[condit]['CMask']
            
            mainfig = plt.figure()
            
            band_idxs = np.where(np.logical_and(self.fvect > dbo.feat_dict[band]['param'][0], self.fvect < dbo.feat_dict[band]['param'][1]))
            if pt == 'all':
                if var_fixed:
                    plot_vect = self.reliablePSD[condit]['BandVect']['Vect']
                else:
                    
                    plot_vect = np.median(self.avgPSD[condit]['PSD'][:,band_idxs].squeeze(),axis=1)
                    #abov_thresh = np.median(self.reliablePSD[condit]['cPSD'][band_idxs,:],axis=1).T.reshape(-1)
                    
            else:
                #just get the patient's diff
                
                plot_vect = np.median(self.feat_diff[pt][condit][:,band_idxs].squeeze(),axis=1)
                
            
            plot_3d_scalp(plot_vect,mainfig,clims=(-3,3),label=condit+label,animate=animate)
            plt.title(condit + ' ' + pt + ' ' + str(var_fixed))
            plt.suptitle(label)
            
            self.write_sig(plot_vect)
            
    def write_sig(self,signature):
        for condit in ['OnT','OffT']:
            #plot_vect = np.zeros((257))
            #plot_vect[self.reliablePSD[condit]['CMask']] = 1
            #plot_vect = self.reliablePSD[condit]['CMask']
            
            #write this to a text file
            np.save('/tmp/' + condit + '_sig.npy',signature)
    
    def plot_diff(self):
        
        for pt in self.pts:
            plt.figure()
            plt.subplot(221)
            plt.plot(self.fvect,self.psd_change[pt]['OnT'].T)
            plt.title('OnT')
            
            plt.subplot(222)
            plt.plot(self.fvect,self.psd_change[pt]['OffT'].T)
            plt.title('OffT')
            
            plt.subplot(223)
            plt.plot(self.fvect,10*np.log10(self.psd_var[pt]['OnT']['BONT'].T))
            
            
            plt.subplot(224)
            plt.plot(self.fvect,10*np.log10(self.psd_var[pt]['OffT']['BOFT'].T))
            
   
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