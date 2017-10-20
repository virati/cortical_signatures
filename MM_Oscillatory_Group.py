# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 06:24:03 2016

@author: virati
"""

import numpy as np
import mne
import DBSOsc
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

import pdb

import scipy.stats as stats

from EEG_Viz import plot_3d_scalp

plt.close('all')
#%%
#Function definitions
#Plot the mean
def plot_PSD_Stats(BONT_result,BOFT_result,sup_add=''):
    plt.figure()
    plt.subplot(211)
    plt.plot(F,BONT_result.T,alpha=0.1)
    plt.axhline(y=0,linewidth=5)
    plt.xlim((0,40))
    plt.ylim((-10,10))
    plt.title('OnTarget EEG PSDs')
    
    plt.subplot(212)
    plt.plot(F,BOFT_result.T,alpha=0.1)
    plt.axhline(y=0,linewidth=5)
    plt.xlim((0,40))
    plt.ylim((-10,10))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power change from PreStim (dB)')
    plt.title('OffTarget EEG PSDs')
    plt.suptitle(pt + sup_add)


def plot_chann_bandpow(BONT_aggr,BOFT_aggr):
    plt.figure()
    plt.subplot(211)
    plt.imshow(BONT_aggr,interpolation='None',aspect='auto')
    plt.yticks(range(len(do_bands)),do_bands)
    plt.xticks(range(len(egipos.ch_names)),egipos.ch_names,rotation='vertical')
    
    plt.subplot(212)
    plt.imshow(BOFT_aggr,interpolation='None',aspect='auto')
    plt.yticks(range(len(do_bands)),do_bands)
    plt.xticks(range(len(egipos.ch_names)),egipos.ch_names,rotation='vertical')
    

def unity(inputvar):
    return inputvar

#%%
def scalp_plotting(BONT_bands,suplabel='',preplot='unity',plot_band=['Alpha']):
    #plt.figure()
    #plt.suptitle(pt + ' ' + suplabel)
    BONT_aggr = []
    BOFT_aggr = []

    #if we want to zscore across all channels before we actually plot do below
    if preplot == 'zscore':
        preplot_f = stats.zscore    
    elif preplot == 'unity':
    #if not do below
        preplot_f = unity
    
    etrodes = egipos.pos
    do_bands = plot_band
    
    for bb,band in enumerate(do_bands):
        #Each band will have its own figure
        #pos_in = mne.channels.create_eeg_layout(egipos.pos)
        #for each band, find the range of values for both BONT and BOFT, so they can be displayed in conjunction with each other
        set_c_max = np.max([BONT_bands[band],BONT_bands[band]])
        set_c_min = np.min([BONT_bands[band],BONT_bands[band]])
        #Instead of the above, just find the cmax for on target for plotting in 3d
        set_c_max = np.max(BONT_bands[band])
        set_c_min = np.min(BONT_bands[band])
        
        norm = mpl.colors.Normalize(vmin=set_c_min,vmax=set_c_max)
        cmap = cm.jet
        m = cm.ScalarMappable(norm=norm,cmap=cmap)
        
        #ax = plt.subplot(2,2,1,projection='3d')
        for ch in range(BONT_bands[band].shape[0]):
            #ax.scatter(etrodes[ch,0],etrodes[ch,1],2*etrodes[ch,2],c=m.to_rgba(BONT_bands[band][ch]),s=50)
            pass
        plot_3d_scalp(BONT_bands[band])
        #plot_3d_scalp(BOFT_bands[band])
        
        
        #ax = plt.subplot(2,2,2)
        #plt.figure()
        #mne.viz.plot_topomap(preplot_f(BONT_bands[band]),pos=egipos.pos[:,[0,1]],vmin=set_c_min,vmax=set_c_max)
        plt.title('BONT Min:' + str(min(BONT_bands[band])) + ' Max:'  + str(max(BONT_bands[band])))

        #plt.figure()
        #mne.viz.plot_topomap(preplot_f(BOFT_bands[band]),pos=egipos.pos[:,[0,1]],vmin=set_c_min,vmax=set_c_max)
        #plt.title('BOFT: ' + str(min(BOFT_bands[band])) + ' Max:'  + str(max(BOFT_bands[band])))
        

        
        #plt.title('On Target ' + band + ' Changes')
        
        #plt.subplot(2,len(do_bands),bb+len(do_bands)+1)
        
        #plt.title('Off Target ' + band + ' Changes')
        
        BONT_aggr.append(BONT_bands[band].T)
        #BOFT_aggr.append(BOFT_bands[band].T)
    
    
    #%%
def topo_plotting(BONT_bands,BOFT_bands,suplabel='',preplot='unity',plot_band=['Alpha']):
    plt.figure()
    plt.suptitle(pt + ' ' + suplabel)
    BONT_aggr = []
    BOFT_aggr = []

    #if we want to zscore across all channels before we actually plot do below
    if preplot == 'zscore':
        preplot_f = stats.zscore    
    elif preplot == 'unity':
    #if not do below
        preplot_f = unity
    
    do_bands = plot_band
        
    for bb,band in enumerate(do_bands):
        #pos_in = mne.channels.create_eeg_layout(egipos.pos)
        #for each band, find the range of values for both BONT and BOFT, so they can be displayed in conjunction with each other
        set_c_max = np.max([BONT_bands[band],BONT_bands[band]])
        set_c_min = np.min([BONT_bands[band],BONT_bands[band]])
        
        plt.subplot(2,len(do_bands),bb+1)
        mne.viz.plot_topomap(preplot_f(BONT_bands[band]),pos=egipos.pos[:,[0,1]],vmin=set_c_min,vmax=set_c_max)
        plt.title('On Target ' + band + ' Changes')
        
        plt.subplot(2,len(do_bands),bb+len(do_bands)+1)
        mne.viz.plot_topomap(preplot_f(BOFT_bands[band]),pos=egipos.pos[:,[0,1]],vmin=set_c_min,vmax=set_c_max)
        plt.title('Off Target ' + band + ' Changes')
        
        #plt.subplot(3,len(do_bands),bb+2*len(do_bands)+1)
        #mne.viz.plot_topomap(preplot_f(BONT_bands[band]),pos=egipos.pos[:,[0,1]],vmin=set_c_min,vmax=set_c_max)
        #plot the 3d version of the BONT
        scalp_plotting(preplot_f(BONT_bands),suplabel='Mean')
        #plt.title(band + '\nOnT')
        
        BONT_aggr.append(BONT_bands[band].T)
        BOFT_aggr.append(BOFT_bands[band].T)
    
def chann_covar(BONT_bands):
    preplot_f = unity
    state = np.array(preplot_f(BONT_bands['Alpha']))
    
    #return the covariance matrix
    return np.cross(state, state)
    
    #plot_chann_bandpow(BONT_aggr,BOFT_aggr)
#%%
#bring the diff matrix from all patients in together
#def big_ideas():
data_in = []
#load in three files
pts = ['905','906','907','908']
BONT_matr = []
BOFT_matr = []

egipos = mne.channels.read_montage('/tmp/GSN-HydroCel-257.sfp')
dzphase = '0Mo'
print('Starting Group Analysis of EEG for timepoint ' + dzphase + ' Experiment: Targeting')

EEG_STATES = defaultdict(dict)

def extract_state(PSD,F):
    state_vect = np.zeros((7,257))
    for bb,band in enumerate(DBSOsc.band_dict.keys()):
        state_vect[bb,:] = DBSOsc.band_pow_raw(PSD,F,band)

    return state_vect

EEG_STATES['BONT'] = defaultdict(dict)
EEG_STATES['BOFT'] = defaultdict(dict)
    
for pp,pt in enumerate(pts):
    input_dict = np.load('/home/virati/MDD_Data/EEG_Sigs_Active/' + dzphase + '/EEG_Sigs_DBS'+pt+ '_' + tpoint + '.npy').item()
    
    #here, instead, we'll take either baseline or stim
    BONT_matr.append((input_dict['BONT']['Total'][:,:,0] - input_dict['BONT']['Total'][:,:,1]))
    BOFT_matr.append((input_dict['BOFT']['Total'][:,:,0] - input_dict['BOFT']['Total'][:,:,1]))
    
    EEG_STATES['BONT'][pt]['PSDiff'] = input_dict['BONT']['Total'][:,:,0] - input_dict['BONT']['Total'][:,:,1]
    EEG_STATES['BOFT'][pt]['PSDiff'] = input_dict['BOFT']['Total'][:,:,0] - input_dict['BOFT']['Total'][:,:,1]
    EEG_STATES[pt]['F'] = input_dict['F']
    
    EEG_STATES['BONT'][pt]['Vect'] = extract_state(EEG_STATES['BONT'][pt]['PSDiff'],EEG_STATES[pt]['F'])
    EEG_STATES['BOFT'][pt]['Vect'] = extract_state(EEG_STATES['BOFT'][pt]['PSDiff'],EEG_STATES[pt]['F'])

#%%    
def plot_topos(condit):
    for pt in pts:
        fig = plt.figure()
        ax = plt.subplot(111)
        plt.title(pt + ' Alpha Topoplot')
        mne.viz.plot_topomap(EEG_STATES[condit][pt]['Vect'][2],pos=egipos.pos[:,[0,1]],vmin=0,vmax=30)

def plot_PSDs(condit):
    for pt in pts:
        fig = plt.figure()
        ax = plt.subplot(111)
        plt.title(pt + ' full PSD')
        plt.plot(EEG_STATES[pt]['F'],EEG_STATES[condit][pt]['PSDiff'].T)
        plt.xlim((0,50))
        
        
#%%
plot_PSDs('BONT')

#%%W
plot_topos('BONT')

#%%
#Redo the PCA with the updates datastructures