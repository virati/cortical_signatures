# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 12:18:37 2016

@author: virati
This script will try to import raw hdEEG data in a continuous way, preprocess it, and generate the figures needed for "Aim 2" - Mapping Cortical Responses/Signatures to Stimulation Parameters
"""

import scipy
import scipy.io as sio
import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import mne

import h5py

from collections import defaultdict

from EEG_Viz import *

plt.close('all')

#def plot_3d_scalp(band):
#    fig = plt.figure()
#    ax = fig.add_subplot(111,projection='3d')
#    egipos = mne.channels.read_montage('/tmp/GSN-HydroCel-257.sfp')
#    etrodes = egipos.pos
#    
#    
#    ax.scatter(etrodes[:,0],etrodes[:,1],10*etrodes[:,2],c=alpha,s=300)
# 
#    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
#    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
#    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
#    # Get rid of the spines                         
#    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0)) 
#    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0)) 
#    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
#    ax.set_xticks([])                               
#    ax.set_yticks([])                               
#    ax.set_zticks([])
#
#    plt.title(pt + ' ' + condit)
#    plt.show()
    

def extract_raw_mat():
    data_dir = '/home/virati/MDD_Data/proc_hdEEG/Continuous/'
    pt_dir = 'DBS906/'
    file = 'DBS906_TurnOn_Day1_Sess1_20150827_024013.mat'
    
    data_dir = '/home/virati/B04/'
    Inp = sio.loadmat(data_dir + pt_dir + file)
    #%%
    
    #Find the key corresponding to the data
    data_key = [key for key in Inp.keys() if key[0:3] == 'DBS']
    
    #%%
    #Data matrix generation
    Data_matr = Inp[data_key[0]]
    
    #%%
    #Spectrogram of the first channel to see
    
    t_bounds = {'Pre_STIM':(760,780), 'BL_STIM':(790,810)}
    t_vect = np.linspace(0,Data_matr.shape[1]/1000,Data_matr.shape[1])
    
    
    signal = defaultdict(dict)
    for ts, tt in t_bounds.items():
        t_loc = np.where(np.logical_and(t_vect > tt[0],t_vect < tt[1]))[0]
        signal[ts] = Inp[data_key[0]][:,t_loc] - np.mean(Inp[data_key[0]][:,t_loc],0)
    
    #Save DataStructure
    sio.savemat('/tmp/test',signal)

def load_raw_mat(fname):
    signal = sio.loadmat(fname)
    
    return signal['EXPORT']['chann'][0][0]

#for condit in ['OnTarget','OffTarget']:
for condit in ['OffTarget']:
    pt = 'DBS907'
    #condit = 'OffTarget'
    
    file = '/home/virati/MDD_Data/proc_hdEEG/' + pt + '/' + pt + '_Sample_Chirp_template/' + pt + '_' + condit + '_all.mat'
    signal = load_raw_mat(fname=file)
    
    def EEG_to_Matr(signal):
        data = []
        
        for ch in range(257):
            data.append(signal[:,ch][0][0][0][0][0])
        data = np.array(data)
        
        return data
    
    #%%
    data = EEG_to_Matr(signal)
    
    mean_sig = np.mean(data,0)
    
    #Re-reference to mean
    for ch in range(257):
        data[ch] = data[ch] - mean_sig
    
    test_dec = sig.decimate(data,10,zero_phase=True)
    plt.plot(test_dec.T)
    #%%
    
    ds_fact = 1
    fs = 500
    epoch = defaultdict(dict)
    nfft=512
    #calculate PSD of each channel
    snippets = {'Baseline':(0,21000),'EarlyStim':(27362,27362+21000)}
    for elabel,ebos in snippets.items():
        P = np.zeros((257,513))
        for ch in range(257):
            sig_filt = sig.decimate(data[ch][ebos[0]:ebos[1]],ds_fact,zero_phase=True)
            #just do a welch estimate
            f,Pxx = sig.welch(sig_filt,fs=fs/ds_fact,window='blackmanharris',nperseg=512,noverlap=128,nfft=2**10)
            
            #First, we're going to go through the timseries, segment it out, and classify each segment in a partial-responsive GMM model
            
            
            #do a spectrogram and then find median
            F,T,SG = sig.spectrogram(sig_filt,nperseg=1024,noverlap=256,window=sig.get_window('blackmanharris',1024),fs=fs/ds_fact)
            Pxx = np.median(SG,axis=1)
            
            P[ch,:] = Pxx
        epoch[elabel] = P
             
    #Compute diff PSD
    diff_PSD = 10*np.log10(epoch['EarlyStim']) - 10*np.log10(epoch['Baseline'])
    #%%
    plt.figure()
    _ = plt.plot(f,diff_PSD.T,alpha=0.2)
    plt.axhline(y=0,linewidth=5)
    #%%
    def plot_ts_chann(data,ch,ds_fact=1):
        plt.figure()
        sel_sig = sig.decimate(data[ch][:],ds_fact,zero_phase=True)
        plt.plot(sel_sig)
    
    #%%
    plt.figure()
    P = epoch['EarlyStim']
    plt.axhline(y=0)
    plt.plot(f,10*np.log10(P.T))
    
    #%%
    #Do a spectrogram of one of the channels
    ch = 5
    sel_sig = sig.decimate(data[ch][:],ds_fact,zero_phase=True)
    plt.figure()
    F,T,SG = sig.spectrogram(sel_sig,nperseg=512,noverlap=256,window=sig.get_window('blackmanharris',512),fs=fs/ds_fact)
    plt.pcolormesh(T,F,10*np.log10(SG))
    
    #take out the alpha band in each channel
    
    alpha = np.zeros((257))
    theta = np.zeros((257))
    for ch in range(257):
        Fidx = np.where(np.logical_and(F>8,F<=14))
        alpha[ch] = np.sum(diff_PSD[ch,Fidx],axis=1)
        Fidx = np.where(np.logical_and(F>4,F<=8))
        theta[ch] = np.sum(diff_PSD[ch,Fidx],axis=1)
    #%%
    #Topoplot it
    
    #egipos = mne.channels.read_montage('/tmp/GSN257.sfp')
    
    set_c_min = np.min(theta)
    set_c_max = np.max(theta)
    #mne.viz.plot_topomap(theta,pos=egipos.pos[3:,[0,1]],vmin=set_c_min,vmax=set_c_max,names=egipos.ch_names[3:],show_names=True,sensors=False)
    #use MNE for 2d plotting
    #mne.viz.plot_topomap(theta,pos=egipos.pos[:,[0,1]],vmin=set_c_min,vmax=set_c_max,names=egipos.ch_names[:],show_names=True,sensors=False)
    
    #Just do a scatter plot
        
    plot_3d_scalp(alpha)
    plt.title(pt + ' ' + condit)
    plt.show()
    plot_3d_scalp(theta)
    plt.title(pt + ' ' + condit)
    plt.show()
    #%%
    snip = (26400,27600)

    
    
    

#%%
#ch = 7

#ts_test = data[:,ch][0][0][0][0][0]

#plt.figure()
#plt.plot(ts_test)


#%%

def MNE_Setup():
    fs = 1000
    ch_names = ["{:02d}".format(x) for x in range(257)]
    
    ch_types = ['eeg'] * 257
    info = mne.create_info(ch_names=ch_names,sfreq=fs,ch_types=ch_types)
    raw = mne.io.RawArray(signal['Pre_STIM'],info)
    event_id = 1
    events = np.array([[200,0,event_id],[1200,0,event_id],[2000,0,event_id]])
    epochs_data = np.array()
    
    #%%
    
    for ts, tt in t_bounds.items():    
        F,T,SG = sig.spectrogram(signal[0,:],nperseg=512,noverlap=256,window=sig.get_window('blackmanharris',512),fs=1000)
        plt.figure()
        plt.pcolormesh(T,F,10*np.log10(SG))
        plt.title(ts)