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
sns.set()
sns.set_style("white")

from DBS_Osc import nestdict

from statsmodels import robust

from sklearn import mixture
from sklearn.decomposition import PCA

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
                    
                    seg_psds = dbo.gen_psd(data_dict,Fs=self.fs,nfft=self.donfft,polyord=polyorder)
                    
                    #gotta flatten the DICTIONARY, so have to do it carefully
                    
                    PSD_matr = np.array([seg_psds[ch] for ch in self.ch_order_list])
                    
                    #find the variance for all segments
                    feat_dict[pt][condit][epoch] = PSD_matr
                    
                    #need to do OSCILLATIONS here
        
        self.feat_dict = feat_dict
    
    def train_simple(self):
        #Train our simple classifier that just finds the shortest distance
        self.signature = {'OnT':0,'OffT':0}
        self.signature['OnT'] = self.pop_osc_change['OnT'][dbo.feat_order.index('Alpha')] / np.linalg.norm(self.pop_osc_change['OnT'][dbo.feat_order.index('Alpha')])
        self.signature['OffT'] = self.pop_osc_change['OffT'][dbo.feat_order.index('Alpha')] / np.linalg.norm(self.pop_osc_change['OffT'][dbo.feat_order.index('Alpha')])
        
    
    def test_simple(self):
        #go to our GMM stack and, for each segment, determine the distance to the two conditions
        
        OnT_sim = nestdict()
        OffT_sim = nestdict()
        
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
    def gen_GMM_dsgn(self,stack_bl=''):
        state_stack = nestdict()
        state_labels = nestdict()
        
        
        for condit in self.condits:
            state_stack[condit] = []
            keyoi = keys_oi[condit][1]
            
            if stack_bl == 'add':
                pt_stacks = [val[condit][keyoi] for pt,val in self.feat_dict.items()] 
                base_len = len(pt_stacks)
                pt_labels = [keyoi] * base_len
                
                pt_stacks += [val[condit]['Off_3'] for pt,val in self.feat_dict.items()]
                pt_labels += ['Off_3'] * (len(pt) - base_len)
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
            
        
        self.GMM_stack = state_stack
        self.GMM_stack_labels = state_labels
    
    def find_seg_covar(self):
        covar_matrix = nestdict()
        
        for condit in self.condits:
            seg_stack = sig.detrend(self.GMM_Osc_stack[condit],axis=1)
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
            
    def pca_decomp(self):
        #check to see if we have what variables we need
        numsegs = self.GMM_Osc_stack['OnT'].shape[1]
        
        Xdsgn = np.mean(self.GMM_Osc_stack['OnT'],axis=1).squeeze()
        
        pca = PCA(n_components=5)
        
        pca.fit(Xdsgn)
        
        self.PCA_d = pca
        self.PCA_x = pca.fit_transform(Xdsgn)
        
        
    
    def gen_GMM_priors(self,condit='OnT'):
        band = 'Alpha'
        
        
        #prior_change = self.pop_osc_mask[condit][dbo.feat_order.index(band)] * self.pop_osc_change[condit][dbo.feat_order.index(band)].reshape(-1,1)
        prior_change = self.pop_osc_change[condit][dbo.feat_order.index(band)].reshape(-1,1)
        
        prior_covar = np.dot(prior_change,prior_change.T)
        
        self.prior_covar = prior_covar
        return prior_covar
        
    
    def gen_GMM_feat(self):
        fvect = self.fvect
        
        feat_out = nestdict()
        
        for condit in self.condits:
            num_segs = self.GMM_stack[condit].shape[1]
            feat_out[condit] = np.zeros((257,num_segs,len(dbo.feat_order)))
            
            for ss in range(num_segs):
                try:
                    feat_out[condit][:,ss,:] = dbo.calc_feats(self.GMM_stack[condit][:,ss,:].squeeze(),fvect).T
                except:
                    pdb.set_trace()
                
        self.GMM_Osc_stack = feat_out
        
    def shape_GMM_dsgn(self):
        segs_feats = nestdict()
        
        for condit in self.condits:
            num_segs = self.GMM_Osc_stack[condit].shape[1]
            
            segs_chann_feats = np.swapaxes(self.GMM_Osc_stack[condit],1,0)
            #this stacks all the ALPHAS for all channels together
            #CHOOSE ALPHA HERE
            segs_feats[condit] = segs_chann_feats[:,:,2]
            #segs_feats = np.reshape(segs_chann_feats,(num_segs,-1),order='F')
            
            #We want a 257 dimensional vector with num_segs observations
        
        return segs_feats

    def pop_meds(self):
        dsgn_X = self.shape_GMM_dsgn()
        X_med = nestdict()
        X_mad = nestdict()
        #do some small simple crap here
        for condit in self.condits:
            X_med[condit]= np.median(dsgn_X[condit],axis=0)
            X_mad[condit] = robust.mad(dsgn_X[condit],axis=0)
        
        self.Seg_Med = (X_med,X_mad)
            
    def train_GMM(self):
        #shape our dsgn matrix properly
        dsgn_X = self.shape_GMM_dsgn()
        

        #let's stack the two together and expect 3 components?
        #this is right shape: (n_samples x n_features)
        
        full_X = np.concatenate([val for key,val in dsgn_X.items()],axis=0)
        
        #setup our covariance prior from OUR OTHER ANALYSES
        
        covariance_prior_alpha = self.gen_GMM_priors()
        
        #when below reg_covar was 1e-1 I got SOMETHING to work
        gMod = mixture.BayesianGaussianMixture(n_components=self.num_gmm_comps,mean_prior=self.Seg_Med[0]['OnT'],mean_precision_prior=0.1,covariance_type='diag',reg_covar=10,tol=1e-2)#,covariance_prior=covariance_prior_alpha)
        #gMod = mixture.BayesianGaussianMixture(n_components=self.num_gmm_comps,covariance_type='full',reg_covar=1,tol=1e-2)#,covariance_prior=covariance_prior_alpha)
        
        try:
            gMod.fit(full_X)
        except:
            pdb.set_trace()
        
        self.GMM = gMod
        
        self.predictions = gMod.predict(full_X)
        self.posteriors = gMod.predict_proba(full_X)
        
        
        
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
            
            avgOsc[condit] = dbo.calc_feats(self.pop_psd_change[condit]['Mean'],self.fvect)
            varOsc[condit] = dbo.calc_feats(self.pop_psd_change[condit]['Var'],self.fvect)
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