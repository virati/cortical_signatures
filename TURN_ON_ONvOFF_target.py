# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 14:18:28 2017

@author: aibrahim
OBSOLETE
"""

import numpy as np
import DBSOsc
from collections import defaultdict
import matplotlib.pyplot as plt

import scipy.stats as stats

#%%
pts = ['906','907','908']
BONT_matr = []
BOFT_matr = []

for pp,pt in enumerate(pts):
    input_dict = np.load('/Users/aibrahim/Documents/Naser/EEG_Sigs_DBS'+pt+'.npy').item()
    
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
    
    do_bands = ['Delta','Theta','Alpha','Beta*','Beta+','Gamma*']
    
    for band in do_bands:
        band_lim = band_dict.returnDict()
        
        f_idx = np.where(np.logical_and(F >= band_lim[band][0], F <= band_lim[band][1]))
    
        #Take the mean power in a band
        BONT_bands_mean[band] = np.mean(np.squeeze(BONT_mean[:,f_idx]),1)
        BOFT_bands_mean[band] = np.mean(np.squeeze(BOFT_mean[:,f_idx]),1)
        
 #%%
    #for our first analysis, we will do a random partition with surrogate data
    #sampling to compare the ON target distribution with the OFF target distribution
    
    #first we will compute correlation coefficient, r, for our ON target 
    #vs OFF target distribution
    a = BONT_bands_mean['Alpha'] - np.mean(BONT_bands_mean['Alpha'])
    b = BOFT_bands_mean['Alpha'] - np.mean(BOFT_bands_mean['Alpha'])
        
    dot_product = sum(a*b)
    r = dot_product/(np.linalg.norm(a)*np.linalg.norm(b))
    
    #let's compute our test statistic, t, from our r value
    t = np.array(r)*(np.sqrt(254/(1-(np.array(r)**2))))
    print(pt+' t = ', t)
    
    #now, we will do our surrogate data testing by combining our Turn on and 6 month
    #distributions, performing a random partition of this data 1000 times, and creating 
    #a histogram of the 1000 t-values we compute
    combined = np.hstack([BONT_bands_mean['Alpha'],BOFT_bands_mean['Alpha']])
    test_stats_hist = []
    
    for i in range(0,1000):
        np.random.shuffle(combined)
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
    #p is the number of test stats more extreme then our t value from our data
    pval = sum(float(num) >= t for num in test_stats_hist) / len(test_stats_hist)
    print(pt+' p = ', pval)
    
    pval = sum(float(num) <= t for num in test_stats_hist) / len(test_stats_hist)
    print(pt+' p = ', pval)
    
    #now lets make a histogram of our surrogate data
    data_range = np.max(test_stats_hist) - np.min(test_stats_hist)
    q25,q75 = np.percentile(test_stats_hist, [25,75])
    Q = q75 - q25
    
    nbins = data_range/(2*Q*(len(test_stats_hist))**(-1/3))
    nbins = int(np.ceil(nbins))    
        
    hist, bin_edges = np.histogram(test_stats_hist,nbins)

    plt.figure()
    plt.bar(bin_edges[:-1],hist,width=.2,edgecolor='black',align='edge')
    plt.xlabel('T-values')
    plt.ylabel('Count')
    plt.title(pt+' Surrogate data Histogram')
    
    surrogate_data = np.array(test_stats_hist)
    
    #%%
    #let's also perform a kullback-liebler divergence to determine how our 
    #turn on and 6 month distributions diverge from one another
    
    #first we need probability distributions (histograms) for both our turn on and 6 month data
    data_range = np.max(BONT_bands_mean['Alpha']) - np.min(BONT_bands_mean['Alpha'])
    q25,q75 = np.percentile(BONT_bands_mean['Alpha'], [25,75])
    Q = q75 - q25
    
    nbins_ON = data_range/(2*Q*(len(BONT_bands_mean['Alpha']))**(-1/3))
    nbins_ON = int(np.ceil(nbins_ON))
    nbins_ON = 25    
        
    hist_ON, bin_edges_ON = np.histogram(BONT_bands_mean['Alpha'],nbins_ON)
    
    p_ON = hist_ON/sum(hist_ON)
    
    plt.figure()
    plt.bar(bin_edges_ON[:-1],hist_ON,width=.1,color='red',edgecolor='black',align='edge')
    plt.xlim(-2.5,2.5)
    plt.xlabel('delta-alpha')
    plt.ylabel('Count')
    plt.title(pt+' ON target delta alpha distribution')
    
    #heres our probability distribution for our OFF target data
    data_range = np.max(BOFT_bands_mean['Alpha']) - np.min(BOFT_bands_mean['Alpha'])
    q25,q75 = np.percentile(BOFT_bands_mean['Alpha'], [25,75])
    Q = q75 - q25
    
    nbins_OFF = data_range/(2*Q*(len(BOFT_bands_mean['Alpha']))**(-1/3))
    nbins_OFF = int(np.ceil(nbins_OFF))    
    nbins_OFF = 25
    
    hist_OFF, bin_edges_OFF = np.histogram(BOFT_bands_mean['Alpha'],nbins_OFF)
    
    p_OFF = hist_OFF/sum(hist_OFF)

    plt.figure()
    plt.bar(bin_edges_OFF[:-1],hist_OFF,width=.1,color='red',edgecolor='black',align='edge')
    plt.xlim(-2.5,2.5)
    plt.xlabel('delta-alpha')
    plt.ylabel('Count')
    plt.title(pt+' OFF target delta alpha distribution')
    
    #%%
    #first, let's calculate DKL(ON target||OFF target)
    #this assumes ON target is the data set, and OFF target is the model
    
    DKL1 = 0
    
    for i in range(0,len(bin_edges_ON)-1):
        pxi = p_ON[i]
        qxi = sum(np.logical_and(BOFT_bands_mean['Alpha'] >= bin_edges_ON[i], BOFT_bands_mean['Alpha'] < bin_edges_ON[i+1]))/sum(hist_OFF)
        if pxi > 0 and qxi > 0:
            val = pxi*np.log10(pxi/qxi)
            DKL1 += val
    print(pt+' DKL(ON target||OFF target) = ', DKL1)
    
    #now, let's calculate DKL(OFF target||ON target)
    #this assumes OFF target is the data set, and ON target is the model
    
    DKL2 = 0
    
    for i in range(0,len(bin_edges_OFF)-1):
        pxi = p_OFF[i]
        qxi = sum(np.logical_and(BONT_bands_mean['Alpha'] >= bin_edges_OFF[i], BONT_bands_mean['Alpha'] < bin_edges_OFF[i+1]))/sum(hist_ON)
        if pxi > 0 and qxi > 0:
            val = pxi*np.log10(pxi/qxi)
            DKL2 += val
    print(pt+' DKL(OFF target||ON target) = ', DKL2)
    
    #%%
    #now lets do a mutual information analysis across the delta alpha distributions
    #and the channel space
    
    #to do thid analysis, first we need to find the joint probability of the 
    #two variables
    
    MI_ON = 0
    
    for j in range(0,257):
        for i in range(0,len(bin_edges_ON)-1):
            px = p_ON[i]
            py = 1/256
            pxy = np.logical_and(BONT_bands_mean['Alpha'][j] >= bin_edges_ON[i], BONT_bands_mean['Alpha'][j] < bin_edges_ON[i+1])/256
            if px > 0 and py > 0 and pxy > 0:
                MIxy = pxy*np.log2(pxy/(px*py))
                MI_ON += MIxy
    print(pt+' MI ON = ', MI_ON)
    
    #lets now statistically test our MI value by shuffling in channel space and 
    #comparing the MI value from data to our distribution
    
    MI_ON_arr = []
    
    for iteration in range(0,100):
        np.random.shuffle(BONT_bands_mean['Alpha'])
        
        MI_ON_shuff = 0
        for j in range(0,257):
            for i in range(0,len(bin_edges_ON)-1):
                px = p_ON[i]
                py = 1/256
                pxy = np.logical_and(BONT_bands_mean['Alpha'][j] >= bin_edges_ON[i], BONT_bands_mean['Alpha'][j] < bin_edges_ON[i+1])/256
                if px > 0 and py > 0 and pxy > 0:
                    MIxy = pxy*np.log2(pxy/(px*py))
                    MI_ON_shuff += MIxy
                    
        MI_ON_arr.append(MI_ON_shuff)
        
    #calculate p-value based on distribution of MI_arr
    #p is the proportion of MI_shuff values more extreme then our MI value from our data
    MI_ON_pval = sum(float(num) >= MI_ON for num in MI_ON_arr) / len(MI_ON_arr)
    print(pt+' MI ON p = ', MI_ON_pval)
    
    MI_ON_pval = sum(float(num) <= MI_ON for num in MI_ON_arr) / len(MI_ON_arr)
    print(pt+' MI ON p = ', MI_ON_pval)
   
    #lets make a histogram of our MI_arr
    data_range = np.max(MI_ON_arr) - np.min(MI_ON_arr)
    q25,q75 = np.percentile(MI_ON_arr, [25,75])
    Q = q75 - q25
    
    nbins_MI_ON = data_range/(2*Q*(len(MI_ON_arr))**(-1/3))
    nbins_MI_ON = int(np.ceil(nbins_MI_ON))    
        
    hist_MI_ON, bin_edges_MI_ON = np.histogram(MI_ON_arr,nbins_MI_ON)
    
    plt.figure()
    plt.bar(bin_edges_MI_ON[:-1],hist_MI_ON,width=.2,color='green',edgecolor='black',align='edge')
    plt.xlabel('Mutual information')
    plt.ylabel('Count')
    plt.title(pt+' ON target Mutual information surrogate data')
    
    #%%
    #Let's get the MI for the off target data and the MI distribtution for the 
    #off target surrogate data
    MI_OFF = 0
    
    for j in range(0,257):
        for i in range(0,len(bin_edges_OFF)-1):
            px = p_OFF[i]
            py = 1/256
            pxy = np.logical_and(BOFT_bands_mean['Alpha'][j] >= bin_edges_OFF[i], BOFT_bands_mean['Alpha'][j] < bin_edges_OFF[i+1])/256
            if px > 0 and py > 0 and pxy > 0:
                MIxy = pxy*np.log2(pxy/(px*py))
                MI_OFF += MIxy
    print(pt+' MI OFF = ', MI_OFF)
    
    #lets now statistically test our MI value by shuffling in channel space and 
    #comparing the MI value from data to our distribution
    
    MI_OFF_arr = []
    
    for iteration in range(0,100):
        np.random.shuffle(BOFT_bands_mean['Alpha'])
        
        MI_OFF_shuff = 0
        for j in range(0,257):
            for i in range(0,len(bin_edges_OFF)-1):
                px = p_OFF[i]
                py = 1/256
                pxy = np.logical_and(BOFT_bands_mean['Alpha'][j] >= bin_edges_OFF[i], BOFT_bands_mean['Alpha'][j] < bin_edges_OFF[i+1])/256
                if px > 0 and py > 0 and pxy > 0:
                    MIxy = pxy*np.log2(pxy/(px*py))
                    MI_OFF_shuff += MIxy
                    
        MI_OFF_arr.append(MI_OFF_shuff)
        
    #calculate p-value based on distribution of MI_arr
    #p is the proportion of MI_shuff values more extreme then our MI value from our data
    MI_OFF_pval = sum(float(num) >= MI_OFF for num in MI_OFF_arr) / len(MI_OFF_arr)
    print(pt+' MI OFF p = ', MI_OFF_pval)
    
    MI_OFF_pval = sum(float(num) <= MI_OFF for num in MI_OFF_arr) / len(MI_OFF_arr)
    print(pt+' MI OFF p = ', MI_OFF_pval)
    
    data_range = np.max(MI_OFF_arr) - np.min(MI_OFF_arr)
    q25,q75 = np.percentile(MI_OFF_arr, [25,75])
    Q = q75 - q25
    
    nbins_MI_OFF = data_range/(2*Q*(len(MI_OFF_arr))**(-1/3))
    nbins_MI_OFF = int(np.ceil(nbins_MI_OFF))    
        
    hist_MI_OFF, bin_edges_MI_OFF = np.histogram(MI_OFF_arr,nbins_MI_OFF)
    
    plt.figure()
    plt.bar(bin_edges_MI_OFF[:-1],hist_MI_OFF,width=.2,color='green',edgecolor='black',align='edge')
    plt.xlabel('Mutual information')
    plt.ylabel('Count')
    plt.title(pt+' OFF target Mutual information surrogate data')
    
#%%
on_data = BONT_bands_mean['Alpha']
off_data = BOFT_bands_mean['Alpha']