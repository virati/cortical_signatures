# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 22:45:28 2016

@author: virati
Description:
This file seeks to be a preliminary analysis script for analysing the cleaned up hdEEG data (all 256 channels) -> doing PCA to find the key channels/bands with maximal changes
"""

#Oscillatory Modes/Signatures of Type1/Type2/White matter stimulation

#Import the matlab file
from scipy.io import loadmat
from collections import defaultdict

import scipy.signal as sig
import numpy as np

import mne

import matplotlib.pyplot as plt

import DBSOsc

plt.close('all')
#%%
def comp_PSD(x_in,nfft=2**11,fs=1000,ds_fact=4):
    #x in is going to be a obs x chann matrix
    P = np.zeros((x_in.shape[0],int(nfft/2+1)))

    #firwin has: # taps, cutoff freq, nyquist freq as arguments    
    filt_lowp = sig.firwin(100,100,nyq=fs/2)
    
    #Go through each channel
    for ii in range(x_in.shape[0]):
        sig_in = x_in[ii,:]
        #Do a lowpass filter at 100Hz
        #sig_filt = sig.filtfilt(filt_lowp,[1],sig_in)
        sig_filt = sig_in
        
        #If input is 1000Hz, and we want to focus on 0-40Hz, we can downsample down to ~100Hz, which is 10x downsampling
        if ds_fact == 1:
            f,P[ii,:] = sig.welch(sig_filt,fs/ds_fact,window='blackmanharris',nperseg=512,noverlap=0,nfft=nfft)
        else:
            f,P[ii,:] = sig.welch(sig.decimate(sig_filt,ds_fact,zero_phase=True),fs/ds_fact,window='blackmanharris',nperseg=512,noverlap=0,nfft=nfft)
        
    
    return f, 10*np.log10(np.abs(P))

#PCA time
def EEG_PCA(in_data):
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(in_data.T)
    
    plt.figure()
    plt.imshow(pca.components_,interpolation='None')


PT = defaultdict(dict)
PT['DBS906'] = defaultdict(dict)
PT['DBS907'] = defaultdict(dict)
PT['DBS908'] = defaultdict(dict)
PT['DBS905'] = defaultdict(dict)
    
tpoint = '0Mo'

#Big Variable with everything
Full_Data = defaultdict(dict)
print('Generating Intermediates for ' + tpoint)

if tpoint == '0Mo':
    PT['DBS906']['BONT']['File'] = '/home/virati/MDD_Data/proc_hdEEG/DBS906/DBS906_AW_Preproc/DBS906_TO_onTAR_MU_HP_LP_seg_mff_cln_ref.mat'
    PT['DBS906']['BOFT']['File'] = '/home/virati/MDD_Data/proc_hdEEG/DBS906/DBS906_AW_Preproc/DBS906_TO_offTAR_bcr_LP_HP_seg_bcr_ref.mat'
    
    PT['DBS907']['BONT']['File'] = '/home/virati/MDD_Data/proc_hdEEG/DBS907/DBS907_TO_onTAR_MU_seg_mff_cln_ref.mat'
    PT['DBS907']['BOFT']['File'] = '/home/virati/MDD_Data/proc_hdEEG/DBS907/DBS907_TO_offTAR_MU_seg_mff_cln_ref.mat'
    
    PT['DBS908']['BONT']['File'] = '/home/virati/MDD_Data/proc_hdEEG/DBS908/DBS908_AW_Preproc/DBS908_TO_onTAR_bcr_LP_seg_mff_cln_ref.mat'
    PT['DBS908']['BOFT']['File'] = '/home/virati/MDD_Data/proc_hdEEG/DBS908/DBS908_AW_Preproc/DBS908_TO_offTAR_bcr_MU_seg_mff_cln_ref.mat'

    PT['DBS905']['BONT']['File'] = '/home/virati/MDD_Data/proc_hdEEG/DBS905/DBS905_AW_Preproc/DBS905_B4_OnTarget_HP_LP_seg_mff_cln_ref_con.mat'
    PT['DBS905']['BOFT']['File'] = '/home/virati/MDD_Data/proc_hdEEG/DBS905/DBS905_AW_Preproc/DBS905_B4_OffTar_HP_LP_seg_mff_cln_ref_con.mat'
elif tpoint == '6Mo':
    PT['DBS906']['BONT']['File'] = '/home/virati/MDD_Data/proc_hdEEG/C24_Targeting/DBS906_C25bio_onTAR_HLP_seg_mff_cln_ref.mat'
    PT['DBS906']['BOFT']['File'] = '/home/virati/MDD_Data/proc_hdEEG/C24_Targeting/DBS906_C25bio_offTAR_HLP_seg_mff_cln_ref.mat'
    
    PT['DBS907']['BONT']['File'] = '/home/virati/MDD_Data/proc_hdEEG/C24_Targeting/DBS907_C25_1_onTAR_HLP_seg_mff_cln_ref.mat'
    PT['DBS907']['BOFT']['File'] = '/home/virati/MDD_Data/proc_hdEEG/C24_Targeting/DBS907_C25_2_offTAR_HLP_seg_mff_cln_ref.mat'
    
    PT['DBS908']['BONT']['File'] = '/home/virati/MDD_Data/proc_hdEEG/C24_Targeting/DBS908_C25_1_onTAR_HLP_seg_mff_cln_ref.mat'
    PT['DBS908']['BOFT']['File'] = '/home/virati/MDD_Data/proc_hdEEG/C24_Targeting/DBS908_C25_2_offTAR_HLP_seg_mff_cln_ref.mat'

    PT['DBS905']['BONT']['File'] = '/home/virati/MDD_Data/proc_hdEEG/C24_Targeting/DBS905_C25_1_onTARGET_20160321_HLP_mff_cln_seg_ref.mat'
    PT['DBS905']['BOFT']['File'] = '/home/virati/MDD_Data/proc_hdEEG/C24_Targeting/DBS905_C25_2_offTarget_HLP_mff_cln_seg_ref.mat'

Stim_Condits = ['BONT','BOFT']
PT_List = ['DBS905','DBS906','DBS907','DBS908']

#%%
#Let's plot all the segments for all patients overlapped
for patient in PT_List:
    for stim_c in CTX.keys():
        plt.figure()
        plt.suptitle(patient + ' ' + stim_c)    
        for ii,intv in enumerate(['STIM','BL']):
            plt.subplot(2,1,ii+1)
            for seg in range(CTX[stim_c][intv].shape[2]):
                plt.plot(CTX[stim_c][intv][:,:,seg].T) #you want transpose for this, for realz; gives you (257,2000).T
                

#%%
#PT_List = ['DBS906']
#patient = 'DBS907'
for patient in PT_List:
    print('Generating Intermediate File for ' + patient)
    CTX = defaultdict(dict)
    for co in Stim_Condits:
        EEG_Data = loadmat(PT[patient][co]['File'])
    
        CTX[co]['STIM'] = EEG_Data[co]
        CTX[co]['BL'] = EEG_Data['Off_3']
    
        del(EEG_Data)
        
    SIGNature = defaultdict(dict)
    State_vect = defaultdict(dict)
    #%%
    #Find mean PSD of all segments
    
    for stim_c in CTX.keys():
        print('Stim Condition: ' + stim_c)
        Pintv = []
        State_vect[stim_c] = defaultdict(dict)
                
        #We loop through every interval we care about for a given condition: STIMulation on and BaseLine
        
        for intv in ['STIM','BL']:
            SIGNature[stim_c] = defaultdict(dict)
            Pseg = []
            alpha_state = []
            delta_state = []
            
            #Here we're going to loop through every segment
            for seg in range(CTX[stim_c][intv].shape[2]):
                f,Pxx = comp_PSD(CTX[stim_c][intv][:,:,seg],ds_fact=1)
                #Doublecheck that the dimensions of the input time series and the output structures are as expected
                
                Pseg.append(Pxx)
                #extract the alpha power for the segment alone
                alpha_state.append(DBSOsc.band_pow_raw(Pxx,f,'Alpha'))
                delta_state.append(DBSOsc.band_pow_raw(Pxx,f,'Delta'))
                #concat the segment timeseries itself
                #segts.append(CTX[stim_c][intv][:,:,seg])
            
            Pseg = np.array(Pseg)
            #Add the median PSD for a given interval/epoch/stim condition to the matrix
            Pintv.append(np.mean(Pseg,0))
            print('Doing ' + patient +  ' ' + intv)
            print(Pseg.shape)
            
            try:
                SIGNature[stim_c][intv]['Segments'] = Pseg.shape[0]
            except:
                print('Segments not the right size')
                
            State_vect[stim_c][intv] = np.vstack((alpha_state,delta_state))
            
        SIGNature[stim_c]['Total'] = np.dstack(Pintv)
        
        SIGNature[stim_c]['IV_States'] = State_vect[stim_c]
        SIGNature[stim_c]['Diff'] = -np.diff(SIGNature[stim_c]['Total'])
    
    SIGNature['F'] = f
    #%%
    
    plt.figure()
    plt.subplot(211)
    for ii in range(257):
        plt.plot(f,SIGNature['BONT']['Diff'][ii,:],alpha=0.1)
    plt.xlim((0,160))
    #plt.ylim((-10,10))
    plt.title('On Target Change from Baseline - PSD: ' + patient)
    
    plt.subplot(212)
    for ii in range(257):
        plt.plot(f,SIGNature['BOFT']['Diff'][ii,:],alpha=0.1)
    plt.xlim((0,160))
    #plt.ylim((-10,10))
    plt.title('Off Target Change from Baseline - PSD: ' + patient)
    
    #Maximally informative (linear)
    in_data = np.squeeze(SIGNature['BOFT']['Diff'] - SIGNature['BONT']['Diff'])
    
    plt.figure()
    plt.plot(f,in_data.T,alpha=0.1)
    plt.xlim((0,160))
    #plt.ylim((-20,20))
    plt.title('On Target Difference from Off Target - PSD: ' + patient)
    plt.suptitle(patient)
    #%%
    np.save('/home/virati/EEG_Sigs_' + patient + '_' + tpoint +  '.npy',SIGNature)
    
    Full_Data[patient] = SIGNature