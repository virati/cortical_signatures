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
import scipy.stats as stats

plt.rcParams['image.cmap'] = 'viridis'
import mne

import h5py

from collections import defaultdict

from DBSpace.visualizations import EEG_Viz as EEG_Viz

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
    
data_dir = '/run/media/virati/Stokes/MDD_Data/hdEEG/Continuous/CHIRPS/'

def extract_raw_mat():
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
for condit in ['OnTarget']:
    pt = 'DBS906'
    #condit = 'OffTarget'
    
    file = data_dir + pt + '_Sample_Chirp_template/' + pt + '_' + condit + '_all.mat'
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
    
    #Decimate down all the data
    test_dec = sig.decimate(data,10,zero_phase=True)
    plt.plot(test_dec.T)
    plt.title('Plotting the decimated Data')
    #%%
    
    ds_fact = 1
    fs = 500
    epoch = defaultdict(dict)
    alpha_t = defaultdict(dict)
    nfft=512
    #calculate PSD of each channel
    snippets = {'Baseline':(0,21000),'EarlyStim':(27362,27362+21000)}
    for elabel,ebos in snippets.items():
        
        #channel x NFFT below
        P = np.zeros((257,257))
        alpha_pow = np.zeros((257,85))
        for ch in range(257):
            sig_filt = sig.decimate(data[ch][ebos[0]:ebos[1]],ds_fact,zero_phase=True)
            #just do a welch estimate
            f,Pxx = sig.welch(sig_filt,fs=fs/ds_fact,window='blackmanharris',nperseg=512,noverlap=128,nfft=2**10)
            
            #First, we're going to go through the timseries, segment it out, and classify each segment in a partial-responsive GMM model
            
            
            #do a spectrogram and then find median
            F,T,SG = sig.spectrogram(sig_filt,nperseg=256,noverlap=10,window=sig.get_window('blackmanharris',256),fs=fs/ds_fact,nfft=512)
            #Take the median along the time axis of the SG to find the median PSD for the epoch label
            Pxx = np.median(SG,axis=1)
            #find timeseries of alpha oscillatory power
            falpha = np.where(np.logical_and(F > 8,F < 14))
            #Frequency is what dimension?? probably 1
            alpha_tcourse = np.median(SG[falpha,:],1)
            
            P[ch,:] = Pxx
            alpha_pow[ch,:] = alpha_tcourse
            
        epoch[elabel] = P
        alpha_t[elabel] = alpha_pow
             
    #Compute diff PSD
    diff_PSD = 10*np.log10(epoch['EarlyStim']) - 10*np.log10(epoch['Baseline'])
    #%%
    plt.figure()
    _ = plt.plot(F,diff_PSD.T,alpha=0.2)
    plt.axhline(y=0,linewidth=5)
    plt.title('Plotting the change in PSD from Baseline to Early Stim')
    #%%
    def plot_ts_chann(data,ch,ds_fact=1):
        plt.figure()
        sel_sig = sig.decimate(data[ch][:],ds_fact,zero_phase=True)
        plt.plot(sel_sig)
    
    #%%
    plt.figure()
    P = epoch['EarlyStim']
    plt.axhline(y=0)
    plt.plot(F,10*np.log10(P.T))
    plt.title('Plotting the Early Stim Epoch Alone')
    
    #%%
    #Do a spectrogram of one of the channels
    ch = [(225,225)]
    for ch1,ch2 in ch:
        if ch1 == ch2:
            sel_sig = sig.decimate(data[ch1][:],ds_fact,zero_phase=True)
        else:
            sel_sig = sig.decimate(data[ch1][:] - data[ch2][:],ds_fact,zero_phase=True)
            #sel_sig = sig.decimate(data[19][:] - np.mean(data[[25,18,11,12,20,26]][:],0),ds_fact,zero_phase=True)
    
    plt.figure()
    F,T,SG = sig.spectrogram(sel_sig,nperseg=512,noverlap=500,window=sig.get_window('blackmanharris',512),fs=fs/ds_fact)
    plt.pcolormesh(T,F,10*np.log10(SG),rasterized=True)
    plt.title('TimeFrequency Signal of Channel ' + str(ch))
    
    #take out sel_sig and sweep the chirp through it
    
    
    #%%
    # focus solely on the ~5Hz power and plot the peak between 50-80 seconds
    t_idxs = np.where(np.logical_and(T > 60,T < 75))
    F_idxs = np.where(np.logical_and(F > 2, F < 10))
    ch_blip = []
    
    for ch in range(257):
        #find the power in 2-6 hertz at 60-65 seconds in
        sel_sig = sig.decimate(data[ch][t_idxs[0]],ds_fact,zero_phase=True)
        _,_,SG = sig.spectrogram(sel_sig,nperseg=512,noverlap=500,window=sig.get_window('blackmanharris',512),fs=fs/ds_fact)
        ch_blip.append(np.max(SG[F_idxs[0],:]) - np.min(SG[F_idxs[0],:]))
    ch_blip = np.array(ch_blip)
    


    #%%
    thresh = 0
    ch_blip_z = stats.zscore(ch_blip)
    plt.hist(ch_blip_z,bins=50,range=(-1,1))
    
    #%%
    EEG_Viz.maya_band_display(ch_blip_z > thresh)
    print(np.where(ch_blip_z > thresh))
    
    #%%
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
        
    #EEG_Viz.plot_3d_scalp(alpha,unwrap=False,scale=100,alpha=0.3,marker_scale=5)
    EEG_Viz.maya_band_display(alpha)
    plt.title(pt + ' ' + condit + ' alpha change')
    plt.show()
    
    EEG_Viz.maya_band_display(theta)
    plt.title(pt + ' ' + condit + ' theta change')
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