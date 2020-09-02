#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 21:31:54 2020

@author: virati

Plot SG of an EEG stream
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 12:18:37 2016

@author: virati
This script will try to import raw hdEEG data in a continuous way, preprocess it, and generate the figures needed for "Aim 2" - Mapping Cortical Responses/Signatures to Stimulation Parameters
THIS IS AN UPDATED FILE NOW SPECIFIC TO 906 until I fix the code to be modular/OOP
"""


import scipy
import scipy.io as sio
import scipy.signal as sig
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats


import mne
import pdb
import h5py

from collections import defaultdict

from DBSpace.visualizations import EEG_Viz as EEG_Viz
plt.rcParams['image.cmap'] = 'jet'
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 30}

matplotlib.rc('font', **font)

plt.close('all')

data_dir = '/run/media/virati/Stokes/MDD_Data/hdEEG/Continuous/ALLMATS/'

file = 'DBS906_TurnOn_Day1_Sess1_20150827_024013.mat'

Inp = sio.loadmat(data_dir + file)
   
    

#Find the key corresponding to the data
data_key = [key for key in Inp.keys() if key[0:3] == 'DBS']

#Spectrogram of the first channel to see
chann = 32
#sg_sig = sig.decimate(Inp[data_key[0]][chann,:],q=10)
sg_sig = Inp[data_key[0]][chann,:]

#do filtering here
sos_lpf = sig.butter(10,20,fs=1000,output='sos')
fsg_sig = sig.sosfilt(sos_lpf,sg_sig)


T,F,SG = sig.spectrogram(sg_sig,nfft=2**10,window='blackmanharris',nperseg=1024,noverlap=500,fs=1000)
fig,ax1 = plt.subplots()
ax1.pcolormesh(F,T,10*np.log10(SG))

ax2 = ax1.twinx()
ax2.plot(np.linspace(0,fsg_sig.shape[0]/1000,fsg_sig.shape[0]),fsg_sig)

#Data matrix generation
Data_matr = Inp[data_key[0]]

#Spectrogram of the first channel to see

#%%
    #%%
    #Do a spectrogram of one of the channels
    ch = [225]
    if len(ch) == 1:
        sel_sig = sig.decimate(data[ch[0]][:],ds_fact,zero_phase=True)
    else:
        sel_sig = sig.decimate(data[ch[0]][:] - data[ch[1]][:],ds_fact,zero_phase=True)
    

    plt.figure()
    F,T,SG = sig.spectrogram(sel_sig,nperseg=512,noverlap=500,window=sig.get_window('blackmanharris',512),fs=fs/ds_fact)
    
    def poly_sub(fVect,psd,order=1):
        polyCoeff = np.polyfit(fVect,10*np.log10(psd),order)
            
        polyfunc = np.poly1d(polyCoeff)
        polyitself = polyfunc(fVect)
        
        
        postpsd = 10**(10*np.log10(psd) - polyitself)
        if (postpsd == 0).any(): raise Exception;
        
        #plt.figure()
        #plt.plot(10*np.log10(psd))
        #plt.plot(polyitself);pdb.set_trace()
        return postpsd
        
    def poly_sub_SG(f,SG):
        post_SG = np.zeros_like(SG)
        for ii in range(SG.shape[1]):
            
            post_SG[:,ii] = poly_sub(f,SG[:,ii])
            
        return post_SG
    #pSG = poly_sub_SG(F,SG)
    
    def norm_SG(f,SG):
        baseline = np.mean(SG[:,0:1000],axis=1)
        plt.plot(baseline)
        post_SG = np.zeros_like(SG)
        for ii in range(SG.shape[1]):
            post_SG[:,ii] = SG[:,ii]/baseline
            
        return post_SG
    nSG = norm_SG(F,SG)
    
    plt.figure()
    plt.pcolormesh(T,F,10*np.log10(nSG),rasterized=True)
    alpha_idxs = np.where(np.logical_and(F < 7,F>2))
    plt.plot(T,10*np.log10(np.mean(nSG[alpha_idxs,:].squeeze(),axis=0)))
    plt.title('TimeFrequency Signal of Channel ' + str(ch))
    plt.figure()
    plt.plot(sel_sig)
    #take out sel_sig and sweep the chirp through it
    
    #%%
    #Do Chirplet Search here
    tvect = np.linspace(0,5,5*1000)
    simil = np.zeros((20,20))
    index = np.zeros((20,20))
    for f0 in range(1,20):
        for f1 in range(1,20):
            chirplet = sig.chirp(tvect,f0=f0,t1=5,f1=f1)
            do_conv = sig.convolve(chirplet,sel_sig)
            simil[f0,f1] = np.max(do_conv)
            index[f0,f1] = int(np.argmax(do_conv))
    
    fig, ax = plt.subplots()

    min_val, max_val, diff = 0., 20., 1.
    
    #imshow portion
    N_points = int((max_val - min_val) / diff)
    imshow_data = np.random.rand(N_points, N_points)
    
    ax.imshow(simil)
    
    #ax.imshow(imshow_data, interpolation='nearest')
    
    #text portion
    ind_array = np.arange(min_val, max_val, diff)
    x, y = np.meshgrid(ind_array, ind_array)
    
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        #pdb.set_trace()
        c = index[int(x_val),int(y_val)]
        ax.text(x_val, y_val, c, va='center', ha='center',size=7)
    
    #set tick marks for grid
    #ax.set_xticks(np.arange(min_val-diff/2, max_val-diff/2))
    #ax.set_yticks(np.arange(min_val-diff/2, max_val-diff/2))
    #ax.set_xticklabels([])
    #ax.set_yticklabels([])
    #ax.set_xlim(min_val-diff/2, max_val-diff/2)
    #ax.set_ylim(min_val-diff/2, max_val-diff/2)
    ax.grid()
    plt.show()

    
    
    
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
    EEG_Viz.maya_band_display(alpha,label=pt + ' ' + condit + ' alpha change')
    #plt.title(pt + ' ' + condit + ' alpha change')
    #plt.show()
    
    EEG_Viz.maya_band_display(theta)
    #plt.title(pt + ' ' + condit + ' theta change')
    #plt.show()
    
    #%%
    snip = (26400,27600)


#%%
#ch = 7

#ts_test = data[:,ch][0][0][0][0][0]

#plt.figure()
#plt.plot(ts_test)


#%%

def OBSMNE_Setup():
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