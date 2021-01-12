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
data_dir = '/home/virati/MDD_Data/hdEEG/Continuous/CHIRPS/'

def extract_raw_mat(fname=[]):
    if fname == []:
        pt_dir = 'DBS906/'
        file = 'DBS906_TurnOn_Day1_Sess1_20150827_024013.mat'
        
        data_dir = '/home/virati/B04/'
        Inp = sio.loadmat(data_dir + pt_dir + file)
    else:
        Inp = sio.loadmat(fname)
        

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
    
    t_bounds = {'Pre_STIM':(760,780), 'BL_STIM':(790,810)}
    t_vect = np.linspace(0,Data_matr.shape[1]/1000,Data_matr.shape[1])
    
    
    signal = defaultdict(dict)
    for ts, tt in t_bounds.items():
        t_loc = np.where(np.logical_and(t_vect > tt[0],t_vect < tt[1]))[0]
        signal[ts] = Inp[data_key[0]][:,t_loc] - np.mean(Inp[data_key[0]][:,t_loc],0)
    
    #Save DataStructure
    sio.savemat('/tmp/test',signal)
#%%
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
    fig,ax = plt.subplots()
    P = epoch['EarlyStim']
    ax.axhline(y=0)
    ax.plot(F,10*np.log10(P.T))
    # Only draw spine between the y-ticks
    ax.spines['left'].set_bounds(-1, 1)
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.title('Plotting the Early Stim Epoch Alone')
    
    #%%
    # Here we'll plot the decimates ts
    # choose a random subsample of channels
    from numpy.random import default_rng
    
    #rand_channs = default_rng().choice(257,size=5,replace=False)
    rand_channs = [32]
    ds_fact=5
    
    decimated_sigs = sig.decimate(data[rand_channs][:],ds_fact,zero_phase=True)
    plt.figure()
    plt.plot(decimated_sigs.T)

#%%
    #%%
    #Do a spectrogram of one of the channels
    ds_fact = 2
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
    plt.pcolormesh(T,F,10*np.log10(SG),rasterized=True)
    alpha_idxs = np.where(np.logical_and(F < 7,F>2))
    plt.plot(T,10*np.log10(np.mean(nSG[alpha_idxs,:].squeeze(),axis=0)))
    plt.title('TimeFrequency Signal of Channel ' + str(ch))
    #%%
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