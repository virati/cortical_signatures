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

#from EEG_Viz import plot_3d_scalp

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

    
def plot_6mo_correlation(on_corr,off_corr):
    plt.figure()
    plt.plot(on_corr)
    plt.plot(off_corr)

    plt.xticks([0,1,2,3,4,5],['Delta','Theta','Alpha','Beta*','Beta+','Gamma*'])
    plt.xlabel('Band')
    plt.ylabel('correlation coeff')
    plt.title(pt + ' 6mo correlation')
    plt.legend(['OnStim','OffStim'])


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
        #plot_3d_scalp(BONT_bands[band])
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
        #scalp_plotting(preplot_f(BONT_bands),suplabel='Mean')
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
pts = ['906','907','908']
BONT_matr = []
BOFT_matr = []

for pp,pt in enumerate(pts):
    input_dict = np.load('/home/virati/MDD_Data/EEG_Sigs_DBS'+pt+'.npy').item()
    #data_in.append(input_dict)
    
    #Here, we're only going to take the DIFFERENCE between the two epochs: baseline and stim
    #BONT_matr.append(input_dict['BONT']['Diff'])
    #BOFT_matr.append(input_dict['BOFT']['Diff'])
    
    #here, instead, we'll take either baseline or stim
    BONT_matr.append((input_dict['BONT']['Total'][:,:,0] - input_dict['BONT']['Total'][:,:,1]))
    BOFT_matr.append((input_dict['BOFT']['Total'][:,:,0] - input_dict['BOFT']['Total'][:,:,1]))
    
    if pt == '906':
        F = input_dict['F']

#Add group element to do all patients
pts = pts + ['GROUP']
#pts = ['GROUP']
var_thresh = 0.5

for pp,pt in enumerate(pts):
    if pt == 'GROUP':
        #take the result across patients in the group
        BONT_mean = np.mean(np.squeeze(np.array(BONT_matr)),0) #indices: patient, channel, PSD, Stim/NoStim (or diff)
        BOFT_mean = np.mean(np.squeeze(np.array(BOFT_matr)),0)
        
        BONT_var = np.std(np.squeeze(np.array(BONT_matr)),0)
        BOFT_var = np.std(np.squeeze(np.array(BOFT_matr)),0)
    else:
        BONT_mean = np.squeeze(np.array(BONT_matr[pp]))
        BOFT_mean = np.squeeze(np.array(BOFT_matr[pp]))
    
    f_idx = []
    
    band_dict = DBSOsc.BandDict()
    
    BONT_bands_mean = defaultdict(dict)
    BOFT_bands_mean = defaultdict(dict)
    BONT_bands_var = defaultdict(dict)
    BOFT_bands_var = defaultdict(dict)
    BONT_bands_consist = defaultdict(dict)
    BOFT_bands_consist = defaultdict(dict)
    
    do_bands = ['Delta','Theta','Alpha','Beta*','Beta+','Gamma*']
    #do_bands = ['Delta','Theta','Alpha','Beta*']
    #do_bands = ['Beta+']
    
    for band in do_bands:
        band_lim = band_dict.returnDict()
        
        f_idx = np.where(np.logical_and(F >= band_lim[band][0], F <= band_lim[band][1]))
    
        #Take the mean power in a band
        BONT_bands_mean[band] = np.mean(np.squeeze(BONT_mean[:,f_idx]),1)
        BOFT_bands_mean[band] = np.mean(np.squeeze(BOFT_mean[:,f_idx]),1)

        #What we actually wanthere is a measure of how "Variance" the channel's power is across PATIENTS
        #This only makes sense in the group average context
        if pt == 'GROUP':
            BONT_bands_var[band] = np.mean(np.squeeze(BONT_var[:,f_idx]),1)
            BOFT_bands_var[band] = np.mean(np.squeeze(BOFT_var[:,f_idx]),1)
            
            BONT_bands_consist[band] = BONT_bands_mean[band] * (1-np.tanh(BONT_bands_var[band]/var_thresh))
            BOFT_bands_consist[band] = BOFT_bands_mean[band] * (1-np.tanh(BOFT_bands_var[band]/var_thresh))
        
        #This doesn't really mean anything here; it just takes the variance within a band, which is kind of dumb
        #BONT_bands_var[band] = np.std(np.squeeze(BONT_mean[:,f_idx]),1)
        #BOFT_bands_var[band] = np.std(np.squeeze(BOFT_mean[:,f_idx]),1)
    
    #%%
    #Plot the stats for the PSDs
    plot_PSD_Stats(BONT_mean,BOFT_mean,sup_add = 'MEAN')
    #plot_PSD_Stats(BONT_var,BOFT_var,sup_add=' STD')

#%%
# shuffle data in the channel space, comment this out when using 6mo data     
    new_idx = list(range(0,257))

    shuffle(new_idx)
    
    BONT_mean_6mo = BONT_mean.T
    BOFT_mean_6mo = BOFT_mean.T

    for ii, sublist in enumerate(BONT_mean_6mo): 
        BONT_mean_6mo[ii] = BONT_mean.T[ii][new_idx]
    
    for ii, sublist in enumerate(BOFT_mean_6mo): 
        BOFT_mean_6mo[ii] = BOFT_mean.T[ii][new_idx]
        
    BONT_mean_6mo = BONT_mean_6mo.T
    BOFT_mean_6mo = BOFT_mean_6mo.T   
        
# create dictionary for band power for 6mo data 
    
    f_idx = []
    
    band_dict = DBSOsc.BandDict()
    
    BONT_bands_mean_6mo = defaultdict(dict)
    BOFT_bands_mean_6mo = defaultdict(dict)
    BONT_bands_var_6mo = defaultdict(dict)
    BOFT_bands_var_6mo = defaultdict(dict)

    
    do_bands = ['Delta','Theta','Alpha','Beta*','Beta+','Gamma*']
    #do_bands = ['Delta','Theta','Alpha','Beta*']
    #do_bands = ['Beta+']
    
    for band in do_bands:
        band_lim = band_dict.returnDict()
        
        f_idx = np.where(np.logical_and(F >= band_lim[band][0], F <= band_lim[band][1]))
    
        #Take the mean power in a band
        BONT_bands_mean_6mo[band] = np.mean(np.squeeze(BONT_mean[:,f_idx]),1)
        BOFT_bands_mean_6mo[band] = np.mean(np.squeeze(BOFT_mean[:,f_idx]),1)

        #What we actually wanthere is a measure of how "Variance" the channel's power is across PATIENTS
        #This only makes sense in the group average context
        if pt == 'GROUP':
            BONT_bands_var_6mo[band] = np.mean(np.squeeze(BONT_var[:,f_idx]),1)
            BOFT_bands_var_6mo[band] = np.mean(np.squeeze(BOFT_var[:,f_idx]),1)

#%%
    plot_PSD_Stats(BONT_mean_6mo,BOFT_mean_6mo,sup_add = 'MEAN')
 
#%%
    egipos = mne.channels.read_montage('/tmp/GSN-HydroCel-257.sfp')
    
    #This plots the topomap
    topo_plotting(BONT_bands_mean,BOFT_bands_mean,suplabel='Mean',plot_band=['Theta'])
    topo_plotting(BONT_bands_mean_6mo,BOFT_bands_mean_6mo,suplabel='6mo Mean',plot_band=['Alpha'])

    #topo_plotting(BONT_bands_mea,BONT_bands_mean-BOFT_bands_mean,suplabel='Mean')

    #this plots the group topomap
    if pt == 'GROUP' and 1:
        topo_plotting(BONT_bands_var,BOFT_bands_var,suplabel='Variance')
        #try something that seems right!
        #I want to mask the mean changes with the variance across patients; to see "what's most consistent" across the patients
        #topo_plotting(BONT_bands_consist,BOFT_bands_consist,suplabel='Mean masked with reliability')

#%%
    # Take dot produt to correlate initial data and 6 mo data
    
    band_corr_on = []
    
    for band in do_bands:
        
        a = BONT_bands_mean[band] - np.mean(BONT_bands_mean[band])
        b = BONT_bands_mean_6mo[band] - np.mean(BONT_bands_mean_6mo[band])
        
        dot_product = sum(a*b)
        corrcoeff = dot_product/(np.linalg.norm(a)*np.linalg.norm(b))
        band_corr_on.append(corrcoeff)
    
    band_corr_off = []
    
    for band in do_bands:
        
        a = BOFT_bands_mean[band] - np.mean(BOFT_bands_mean[band])
        b = BOFT_bands_mean_6mo[band] - np.mean(BOFT_bands_mean_6mo[band])
        
        dot_product = sum(a*b)
        corrcoeff = dot_product/(np.linalg.norm(a)*np.linalg.norm(b))
        band_corr_off.append(corrcoeff)
        
    print('Band correlation coefficients = ', band_corr_on)

    #calulate t-statistic and p-values from correlation coefficients
    
    tvals = np.array(band_corr_on)*(np.sqrt(254/(1-(np.array(band_corr_on)**2))))
    
    pval = stats.t.sf(np.abs(tvals), 254)*2
    
    print('\n')
    print('T-values = ', tvals, ' p-values = ', pval)
        
#%%
    plot_6mo_correlation(band_corr_on,band_corr_off)        

#%% 
# Create a histogram for surrogate data testing for Alpha only
#ultimately we will shuffle initial and 6 mo data, create two random subsets and calculate corr coeff
#this will be done 1000 to test statistical significance of correlation of actual data

    combined = np.hstack([BONT_bands_mean['Alpha'],BONT_bands_mean_6mo['Alpha']])
    test_stats_hist = []
    
    # we will iterate over 1000 permutations of random subsets to create a histogram of test statistics
    for i in range(0,1000):
        shuffle(combined)
        subset1 = combined[:257]
        subset2 = combined[257:]
        
        #take the dot product of the subsets
        a = subset1 - np.mean(subset1)
        b = subset2 - np.mean(subset2)
        
        dot_product = sum(a*b)
        corrcoeff = dot_product/(np.linalg.norm(a)*np.linalg.norm(b))
        test_stat = corrcoeff*(np.sqrt(254/(1-(corrcoeff**2))))
        
        test_stats_hist.append(test_stat)
        
    #calculate p-value based on distribution of test statistics
    alpha_band_test_stat = tvals[2]
    pval = sum(float(num) >= alpha_band_test_stat for num in test_stats_hist) / len(test_stats_hist)
    print('p = ', pval)
        
    #calculate number of bins for histogram of test stats

    data_range = np.max(test_stats_hist) - np.min(test_stats_hist)
    q25,q75 = np.percentile(test_stats_hist, [25,75])
    Q = q75 - q25
    
    nbins = data_range/(2*Q*(len(test_stats_hist))**(-1/3))
    nbins = int(np.ceil(nbins))    
    
    #compute x values (bins) and y values (probabilty of each bin)
    
    hist, bin_edges = np.histogram(test_stats_hist,nbins)

    plt.figure()
    plt.bar(bin_edges[:-1],hist,width=.2,edgecolor='black',align='edge')
    plt.xlabel('T-values')
    plt.ylabel('Count')
    plt.title('Histogram')
    
    #entropy of the system
    
    p = hist/sum(hist)
    
    entropy = 0
    
    for i in range(nbins):
        if p[i] > 0:
            val = p[i]*np.log2(p[i])
            entropy += val
        else:
           entropy = entropy
        
    entropy = -entropy
    
    print(entropy, ' bits')

#%%    
scalp_plotting(BONT_bands_mean,suplabel='Mean')

#%%
#let's do PCA on the full channel-space data
ch_sig_matr = np.zeros((6,257))

for bb, band in enumerate(do_bands):
    ch_sig_matr[bb,:] = BONT_bands_mean[band]

ch_sig_matr = ch_sig_matr[:4,:]
#do simple PCA of this
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(ch_sig_matr.T)

PC1 = pca.components_
varexpl = pca.explained_variance_ratio_
rotdata = pca.transform(ch_sig_matr.T)

#first component
for ii in range(4):
    plt.figure()
    mne.viz.plot_topomap(rotdata[:,ii],pos=egipos.pos[:,[0,1]])
    plt.title(str(ii) + ' Component: '  + str(PC1[ii,:]))
    plt.show()
#%%

plt.plot(PC1.T)
plt.legend(['PC1','PC2','PC3','PC4'])
plt.xticks([0,1,2,3],['Delta','Theta','Alpha','Beta*'])
plt.show()

plt.plot(varexpl)
plt.xticks([0,1,2,3],['PC1','PC2','PC3','PC4'])

#%%
#Do ICA instead
from sklearn.decomposition import FastICA
rng = np.random.RandomState(42)

ica = FastICA(random_state=rng)
Sica = ica.fit(ch_sig_matr.T)
comps = Sica.components_

rot_ica = ica.transform(ch_sig_matr.T)

for ii in range(4):
    plt.figure()
    mne.viz.plot_topomap(rot_ica[:,ii],pos=egipos.pos[:,[0,1]])
    plt.show()
    
plt.plot(comps.T)
plt.legend(['1','2','3','4'])
plt.show()

#%%
scalp_plotting([],suplabel='Empty')
#%%

#do pairwise coherence for all channels in Alpha
alpha_covar = chann_covar(BONT_bands_mean)

#%%
#        
#    #try 3d plotting
#    #convert from PSDs to timeseries
#    chtyp = ['eeg'] * 257
#    chnam = egipos.ch_names
#    
#    info = mne.create_info(chnam,1000,chtyp)
#    ts = np.zeros((513,257))
#    for cc in range(257):
#        ts[:,cc] = np.fft.ifft(BONT_mean[cc,1025/2:])
#    
#    data = ts[513/2:,:].T
#    agr = mne.io.RawArray(data,info)
#    evok = mne.EvokedArray(data,info,tmin=0)
#    _ = evok.plot()
#    
    #fmap = mne.make_field_map(evok,subject='Sample')
    #field = evok.plot_field(fmap,time=0)
        
