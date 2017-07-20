# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 10:09:09 2017

@author: aibrahim
"""

import numpy as np
import mne
import DBSOsc
from collections import defaultdict
import matplotlib.pyplot as plt


#%%
def unity(inputvar):
    return inputvar

    #%%
def topo_plotting(BONT_bands,BOFT_bands,suplabel='',preplot='unity',plot_band=['Alpha']):
    plt.figure()
    plt.suptitle(' ' + suplabel)
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
        set_c_max = np.max([BONT_bands,BONT_bands])
        set_c_min = np.min([BONT_bands,BONT_bands])
        
        plt.subplot(2,len(do_bands),bb+1)
        mne.viz.plot_topomap(preplot_f(BONT_bands),pos=egipos.pos[:,[0,1]],vmin=set_c_min,vmax=set_c_max)
        
        plt.subplot(2,len(do_bands),bb+len(do_bands)+1)
        mne.viz.plot_topomap(preplot_f(BOFT_bands),pos=egipos.pos[:,[0,1]],vmin=set_c_min,vmax=set_c_max)
        
        BONT_aggr.append(BONT_bands.T)
        BOFT_aggr.append(BOFT_bands.T)


#%%
pts = ['906','907','908']
BONT_matr = []
BOFT_matr = []

BONT_matr_6mo = []
BOFT_matr_6mo = []

for pp,pt in enumerate(pts):
    input_dict = np.load('/Users/aibrahim/Documents/Naser/0moEEG_Sigs_DBS'+pt+'.npy').item()
    
    #here, instead, we'll take either baseline or stim
    BONT_matr.append((input_dict['BONT']['Total'][:,:,0] - input_dict['BONT']['Total'][:,:,1]))
    BOFT_matr.append((input_dict['BOFT']['Total'][:,:,0] - input_dict['BOFT']['Total'][:,:,1]))
    
    input_dict_6mo = np.load('/Users/aibrahim/Documents/Naser/6moEEG_Sigs_DBS'+pt+'.npy').item()
    
    BONT_matr_6mo.append((input_dict_6mo['BONT']['Total'][:,:,0] - input_dict_6mo['BONT']['Total'][:,:,1]))
    BOFT_matr_6mo.append((input_dict_6mo['BOFT']['Total'][:,:,0] - input_dict_6mo['BOFT']['Total'][:,:,1]))
    
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

    do_bands = ['Delta','Theta','Alpha','Beta*','Beta+','Gamma*']
    
    for band in do_bands:
        band_lim = band_dict.returnDict()
        
        f_idx = np.where(np.logical_and(F >= band_lim[band][0], F <= band_lim[band][1]))
    
        #Take the mean power in a band
        BONT_bands_mean[band] = np.mean(np.squeeze(BONT_mean[:,f_idx]),1)
        BOFT_bands_mean[band] = np.mean(np.squeeze(BOFT_mean[:,f_idx]),1)
    
    #%% now we'll do the same thing with the 6 month data
    if pt == 'GROUP':
        #take the result across patients in the group
        BONT_mean_6mo = np.mean(np.squeeze(np.array(BONT_matr_6mo)),0) #indices: patient, channel, PSD, Stim/NoStim (or diff)
        BOFT_mean_6mo = np.mean(np.squeeze(np.array(BOFT_matr_6mo)),0)
        
        BONT_var_6mo = np.std(np.squeeze(np.array(BONT_matr_6mo)),0)
        BOFT_var_6mo = np.std(np.squeeze(np.array(BOFT_matr_6mo)),0)
    else:
        BONT_mean_6mo = np.squeeze(np.array(BONT_matr_6mo[pp]))
        BOFT_mean_6mo = np.squeeze(np.array(BOFT_matr_6mo[pp]))
    
    f_idx = []
    
    band_dict = DBSOsc.BandDict()
    
    BONT_bands_mean_6mo = defaultdict(dict)
    BOFT_bands_mean_6mo = defaultdict(dict)
    
    do_bands = ['Delta','Theta','Alpha','Beta*','Beta+','Gamma*']
    
    for band in do_bands:
        band_lim = band_dict.returnDict()
        
        f_idx = np.where(np.logical_and(F >= band_lim[band][0], F <= band_lim[band][1]))
    
        #Take the mean power in a band
        BONT_bands_mean_6mo[band] = np.mean(np.squeeze(BONT_mean_6mo[:,f_idx]),1)
        BOFT_bands_mean_6mo[band] = np.mean(np.squeeze(BOFT_mean_6mo[:,f_idx]),1)

    #change this depending on what band we want to test
    data = BONT_bands_mean['Alpha']
    model1 = BONT_bands_mean_6mo['Alpha']
    model2 = BOFT_bands_mean['Alpha']
    #%%
    egipos = mne.channels.read_montage('/Users/aibrahim/Documents/Naser/GSN-HydroCel-257.sfp')
    
    topo_plotting(data,data,suplabel='Data',plot_band=['Alpha'])
    #This plots the topomap
    topo_plotting(model1,model2,suplabel='Models',plot_band=['Alpha'])
    
    #%%
    #here's our probability distribution of our data
    data_range = np.max(data) - np.min(data)
    q25,q75 = np.percentile(data, [25,75])
    Q = q75 - q25
    
    nbins_data = data_range/(2*Q*(len(data))**(-1/3))
    nbins_data = int(np.ceil(nbins_data))    
        
    hist_data, bin_edges_data = np.histogram(data,nbins_data)
    
    p_data = hist_data/sum(hist_data)
    
    #heres our probability distribution for our model1
    data_range = np.max(model1) - np.min(model1)
    q25,q75 = np.percentile(model1, [25,75])
    Q = q75 - q25
    
    nbins_model1 = data_range/(2*Q*(len(model1))**(-1/3))
    nbins_model1 = int(np.ceil(nbins_model1))    
        
    hist_model1, bin_edges_model1 = np.histogram(model1,nbins_model1)
    
    p_model1 = hist_model1/sum(hist_model1)
    
    #heres our probability distribution for our OFF target data
    data_range = np.max(model2) - np.min(model2)
    q25,q75 = np.percentile(model2, [25,75])
    Q = q75 - q25
    
    nbins_model2 = data_range/(2*Q*(len(model2))**(-1/3))
    nbins_model2 = int(np.ceil(nbins_model2))    
        
    hist_model2, bin_edges_model2 = np.histogram(model2,nbins_model2)
    
    p_model2 = hist_model2/sum(hist_model2)
    
    #%%
    #first, let's calculate DKL(data||model1)
    
    DKL1 = 0
    
    for i in range(0,len(bin_edges_data)-1):
        pxi = p_data[i]
        qxi = sum(np.logical_and(model1 >= bin_edges_data[i], model1 < bin_edges_data[i+1]))/sum(hist_model1)
        if pxi > 0 and qxi > 0:
            val = pxi*np.log10(pxi/qxi)
            DKL1 += val
    print(pt+' DKL(data||model1) = ', DKL1)
    
    #now, let's calculate DKL(data||model2)
    
    DKL2 = 0
    
    for i in range(0,len(bin_edges_data)-1):
        pxi = p_data[i]
        qxi = sum(np.logical_and(model2 >= bin_edges_data[i], model2 < bin_edges_data[i+1]))/sum(hist_model2)
        if pxi > 0 and qxi > 0:
            val = pxi*np.log10(pxi/qxi)
            DKL2 += val
    print(pt+' DKL(data||model2) = ', DKL2)
    
    #%%
    # let's do a likelihood ratio test
    
    #first, let's find probability density function of data under the alternative hypothesis,
    #which we will assume is that the data is closer to model 1
    LR = np.log10(DKL2/DKL1)
    print(pt+' LR = ', LR)
    
    #Let's find the expected likelihood ratio if the alternative is true
    # EA lambda = DKL(A||H)
    # DKL(model 1||model 2)
    
    DKL_alternative = 0
    
    for i in range(0,len(bin_edges_model1)-1):
        pxi = p_model1[i]
        qxi = sum(np.logical_and(model2 >= bin_edges_model1[i], model2 < bin_edges_model1[i+1]))/sum(hist_model2)
        if pxi > 0 and qxi > 0:
            val = pxi*np.log10(pxi/qxi)
            DKL_alternative += val
    print(pt+' DKL(model 1||model 2) = ', DKL_alternative)
    
    #now we'll find the expected likelihood ratio if the null is true
    # EH lambda = -DKL(H||A)
    # -DKL(model 2||model 1)
    
    DKL_null = 0
    
    for i in range(0,len(bin_edges_model2)-1):
        pxi = p_model2[i]
        qxi = sum(np.logical_and(model1 >= bin_edges_model2[i], model1 < bin_edges_model2[i+1]))/sum(hist_model1)
        if pxi > 0 and qxi > 0:
            val = pxi*np.log10(pxi/qxi)
            DKL_null += val
    print(pt+' DKL(model 2||model 1) = ', -DKL_null)
    
    #%%
    #lets do a random partition test using chi squared as our test statistic to see 
    #if ON target at TURN ON and at 6 months are from significantly different distributions
        
    chi = 0
    
    for i in range(0,len(bin_edges_data)-1):
        Ri = hist_data[i]
        Si = sum(np.logical_and(model1 >= bin_edges_data[i], model1 < bin_edges_data[i+1]))
        if Ri > 0 and Si > 0:
            val = ((Ri-Si)**2)/(Ri+Si)
            chi += val
            
    #%%
    #now, we will do our surrogate data testing by combining our Turn on and 6 month
    #distributions, performing a random partition of this data 1000 times, and creating 
    #a histogram of the 1000 t-values we compute
    combined = np.hstack([data,model1])
    test_stats_hist_chi = []
    
    for i in range(0,1000):
        np.random.shuffle(combined)
        subset1 = combined[:257]
        subset2 = combined[257:]
        
        #%%
        #now lets get our distributions to get chi and DKL histograms for the shuffled data
        data_range = np.max(subset1) - np.min(subset1)
        q25,q75 = np.percentile(subset1, [25,75])
        Q = q75 - q25
        
        nbins_subset1 = data_range/(2*Q*(len(subset1))**(-1/3))
        nbins_subset1 = int(np.ceil(nbins_subset1))    
        
        hist_subset1, bin_edges_subset1 = np.histogram(subset1,nbins_subset1)
        
        p_subset1 = hist_subset1/sum(hist_subset1)
        
        #heres our probability distribution for our model1
        data_range = np.max(subset2) - np.min(subset2)
        q25,q75 = np.percentile(subset2, [25,75])
        Q = q75 - q25
        
        nbins_subset2 = data_range/(2*Q*(len(subset2))**(-1/3))
        nbins_subset2 = int(np.ceil(nbins_subset2))    
            
        hist_subset2, bin_edges_subset2 = np.histogram(subset2,nbins_subset2)
        
        p_subset2 = hist_subset2/sum(hist_subset2)  
        
        #%%
        #now lets get our histogram of chi values
        chi_shuff = 0
    
        for i in range(0,len(bin_edges_subset1)-1):
            Ri = hist_subset1[i]
            Si = sum(np.logical_and(subset2 >= bin_edges_subset1[i], subset2 < bin_edges_subset1[i+1]))
            if Ri > 0 and Si > 0:
                val = ((Ri-Si)**2)/(Ri+Si)
                chi_shuff += val
            
        test_stats_hist_chi.append(chi_shuff)

        
    #%%
    #calculate p-value based on distribution of test statistics
    #p is the number of test stats more extreme then our t value from our data

    pval1_chi = sum(float(num) >= chi for num in test_stats_hist_chi) / len(test_stats_hist_chi)
    pval2_chi = sum(float(num) <= chi for num in test_stats_hist_chi) / len(test_stats_hist_chi)
    
    print('\n')
    print(pt+' chi = ', chi)
    print(pt+' p = ',pval1_chi,' ',pval2_chi)

    
    data_range = np.max(test_stats_hist_chi) - np.min(test_stats_hist_chi)
    q25,q75 = np.percentile(test_stats_hist_chi, [25,75])
    Q = q75 - q25
    
    nbins_chi = data_range/(2*Q*(len(test_stats_hist_chi))**(-1/3))
    nbins_chi = int(np.ceil(nbins_chi))    
        
    hist_chi, bin_edges_chi = np.histogram(test_stats_hist_chi,nbins_chi)
    
    plt.figure()
    plt.bar(bin_edges_chi[:-1],hist_chi,width=1,edgecolor='black',align='edge')
    plt.xlabel('Chi squared values')
    plt.ylabel('Count')
    plt.title(pt+' Turn on vs 6 months surrogate data')