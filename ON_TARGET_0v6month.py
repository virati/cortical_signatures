# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 14:43:27 2017

@author: aibrahim
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

    #%%
    #for our first analysis, we will do a random partition with surrogate data
    #sampling to compare the Turn on distribution with the 6 month distribution
    
    #first we will compute correlation coefficient, r, for our ON target distributions
    #at turn on and at 6 months
    a = BONT_bands_mean['Alpha'] - np.mean(BONT_bands_mean['Alpha'])
    b = BONT_bands_mean_6mo['Alpha'] - np.mean(BONT_bands_mean_6mo['Alpha'])
        
    dot_product = sum(a*b)
    r = dot_product/(np.linalg.norm(a)*np.linalg.norm(b))
    print(pt+' r = ', r)
    
    #let's compute our test statistic, t, from our r value
    t = np.array(r)*(np.sqrt(254/(1-(np.array(r)**2))))
    print(pt+' t = ', t)
    
    #now, we will do our surrogate data testing by combining our Turn on and 6 month
    #distributions, performing a random partition of this data 1000 times, and creating 
    #a histogram of the 1000 t-values we compute
    combined = np.hstack([BONT_bands_mean['Alpha'],BONT_bands_mean_6mo['Alpha']])
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
    
    #%%
    #let's also perform a kullback-liebler divergence to determine how our 
    #turn on and 6 month distributions diverge from one another
    
    #first we need probability distributions (histograms) for both our turn on and 6 month data
    data_range = np.max(BONT_bands_mean['Alpha']) - np.min(BONT_bands_mean['Alpha'])
    q25,q75 = np.percentile(BONT_bands_mean['Alpha'], [25,75])
    Q = q75 - q25
    
    nbins_0mo = data_range/(2*Q*(len(BONT_bands_mean['Alpha']))**(-1/3))
    nbins_0mo = int(np.ceil(nbins_0mo))    
        
    hist_0mo, bin_edges_0mo = np.histogram(BONT_bands_mean['Alpha'],nbins_0mo)
    
    p_0mo = hist_0mo/sum(hist_0mo)

    plt.figure()
    plt.bar(bin_edges_0mo[:-1],hist_0mo,width=.2,color='red',edgecolor='black',align='edge')
    plt.xlim(-2.5,2.5)
    plt.xlabel('delta-alpha')
    plt.ylabel('Count')
    plt.title(pt+' TURN ON on target delta alpha distribution')
    
    #heres our probability distribution for our 6 month data
    data_range = np.max(BONT_bands_mean_6mo['Alpha']) - np.min(BONT_bands_mean_6mo['Alpha'])
    q25,q75 = np.percentile(BONT_bands_mean_6mo['Alpha'], [25,75])
    Q = q75 - q25
    
    nbins_6mo = data_range/(2*Q*(len(BONT_bands_mean_6mo['Alpha']))**(-1/3))
    nbins_6mo = int(np.ceil(nbins_6mo))    
        
    hist_6mo, bin_edges_6mo = np.histogram(BONT_bands_mean_6mo['Alpha'],nbins_6mo)
    
    p_6mo = hist_6mo/sum(hist_6mo)

    plt.figure()
    plt.bar(bin_edges_6mo[:-1],hist_6mo,width=.2,color='red',edgecolor='black',align='edge')
    plt.xlim(-2.5,2.5)
    plt.xlabel('delta-alpha')
    plt.ylabel('Count')
    plt.title(pt+' 6 MONTHS on target delta alpha distribution')
    
    #%%
    #first, let's calculate DKL(On target TURN ON||On target 6mo)
    #this assumes Turn on is the data set, and 6 mo is the model
    
    DKL1 = 0
    
    for i in range(0,len(bin_edges_0mo)-1):
        pxi = p_0mo[i]
        qxi = sum(np.logical_and(BONT_bands_mean_6mo['Alpha'] >= bin_edges_0mo[i], BONT_bands_mean_6mo['Alpha'] < bin_edges_0mo[i+1]))/sum(hist_6mo)
        if pxi > 0 and qxi > 0:
            val = pxi*np.log10(pxi/qxi)
            DKL1 += val
    print(pt+' DKL(On target TURN ON||On target 6 months) = ', DKL1)
    
    #now, let's calculate DKL(On target 6mo||On target TURN ON)
    #this assumes 6 months is the data set, and turn on is the model
    
    DKL2 = 0
    
    for i in range(0,len(bin_edges_6mo)-1):
        pxi = p_6mo[i]
        qxi = sum(np.logical_and(BONT_bands_mean['Alpha'] >= bin_edges_6mo[i], BONT_bands_mean['Alpha'] < bin_edges_6mo[i+1]))/sum(hist_0mo)
        if pxi > 0 and qxi > 0:
            val = pxi*np.log10(pxi/qxi)
            DKL2 += val
    print(pt+' DKL(On target 6 months||On target TURN ON) = ', DKL2)
    
    #%%
    #now lets do a mutual information analysis across the delta alpha distributions
    #and the channel space
    
    #to do thid analysis, first we need to find the joint probability of the 
    #two variables
    
    #lets do this analysis for our turn on data first
    
    MI_0mo = 0
    
    for j in range(0,257):
        for i in range(0,len(bin_edges_0mo)-1):
            px = p_0mo[i]
            py = 1/256
            pxy = np.logical_and(BONT_bands_mean['Alpha'][j] >= bin_edges_0mo[i], BONT_bands_mean['Alpha'][j] < bin_edges_0mo[i+1])/256
            if px > 0 and py > 0 and pxy > 0:
                MIxy = pxy*np.log2(pxy/(px*py))
                MI_0mo += MIxy
    print(pt+' MI TURN ON = ', MI_0mo)
    
    #lets now statistically test our MI value by shuffling in channel space and 
    #comparing the MI value from data to our distribution
    
    MI_0mo_arr = []
    
    for iteration in range(0,100):
        np.random.shuffle(BONT_bands_mean['Alpha'])
        
        MI_0mo_shuff = 0
        for j in range(0,257):
            for i in range(0,len(bin_edges_0mo)-1):
                px = p_0mo[i]
                py = 1/256
                pxy = np.logical_and(BONT_bands_mean['Alpha'][j] >= bin_edges_0mo[i], BONT_bands_mean['Alpha'][j] < bin_edges_0mo[i+1])/256
                if px > 0 and py > 0 and pxy > 0:
                    MIxy = pxy*np.log2(pxy/(px*py))
                    MI_0mo_shuff += MIxy
                    
        MI_0mo_arr.append(MI_0mo_shuff)
        
    #calculate p-value based on distribution of MI_arr
    #p is the proportion of MI_shuff values more extreme then our MI value from our data
    MI_0mo_pval = sum(float(num) >= MI_0mo for num in MI_0mo_arr) / len(MI_0mo_arr)
    print(pt+' MI TURN ON p = ', MI_0mo_pval)
    
    MI_0mo_pval = sum(float(num) <= MI_0mo for num in MI_0mo_arr) / len(MI_0mo_arr)
    print(pt+' MI TURN ON p = ', MI_0mo_pval)
   
    #lets make a histogram of our MI_arr
    data_range = np.max(MI_0mo_arr) - np.min(MI_0mo_arr)
    q25,q75 = np.percentile(MI_0mo_arr, [25,75])
    Q = q75 - q25
    
    nbins_MI_0mo = data_range/(2*Q*(len(MI_0mo_arr))**(-1/3))
    nbins_MI_0mo = int(np.ceil(nbins_MI_0mo))    
        
    hist_MI_0mo, bin_edges_MI_0mo = np.histogram(MI_0mo_arr,nbins_MI_0mo)
    
    plt.figure()
    plt.bar(bin_edges_MI_0mo[:-1],hist_MI_0mo,width=.2,color='green',edgecolor='black',align='edge')
    plt.xlabel('Mutual information')
    plt.ylabel('Count')
    plt.title(pt+' TURN ON on target Mutual information surrogate data')
    
    #%%
    #Let's get the MI for the 6 month data and the MI distribtution for the 
    #6 month surrogate data
    MI_6mo = 0
    
    for j in range(0,257):
        for i in range(0,len(bin_edges_6mo)-1):
            px = p_6mo[i]
            py = 1/256
            pxy = np.logical_and(BONT_bands_mean_6mo['Alpha'][j] >= bin_edges_6mo[i], BONT_bands_mean_6mo['Alpha'][j] < bin_edges_6mo[i+1])/256
            if px > 0 and py > 0 and pxy > 0:
                MIxy = pxy*np.log2(pxy/(px*py))
                MI_6mo += MIxy
    print(pt+' MI 6 months = ', MI_6mo)
    
    #lets now statistically test our MI value by shuffling in channel space and 
    #comparing the MI value from data to our distribution
    
    MI_6mo_arr = []
    
    for iteration in range(0,100):
        np.random.shuffle(BONT_bands_mean_6mo['Alpha'])
        
        MI_6mo_shuff = 0
        for j in range(0,257):
            for i in range(0,len(bin_edges_6mo)-1):
                px = p_6mo[i]
                py = 1/256
                pxy = np.logical_and(BONT_bands_mean_6mo['Alpha'][j] >= bin_edges_6mo[i], BONT_bands_mean_6mo['Alpha'][j] < bin_edges_6mo[i+1])/256
                if px > 0 and py > 0 and pxy > 0:
                    MIxy = pxy*np.log2(pxy/(px*py))
                    MI_6mo_shuff += MIxy
                    
        MI_6mo_arr.append(MI_6mo_shuff)
        
    #calculate p-value based on distribution of MI_arr
    #p is the proportion of MI_shuff values more extreme then our MI value from our data
    MI_6mo_pval = sum(float(num) >= MI_6mo for num in MI_6mo_arr) / len(MI_6mo_arr)
    print(pt+' MI 6 months p = ', MI_6mo_pval)
    
    MI_6mo_pval = sum(float(num) <= MI_6mo for num in MI_6mo_arr) / len(MI_6mo_arr)
    print(pt+' MI 6 months p = ', MI_6mo_pval)
    
    data_range = np.max(MI_6mo_arr) - np.min(MI_6mo_arr)
    q25,q75 = np.percentile(MI_6mo_arr, [25,75])
    Q = q75 - q25
    
    nbins_MI_6mo = data_range/(2*Q*(len(MI_6mo_arr))**(-1/3))
    nbins_MI_6mo = int(np.ceil(nbins_MI_6mo))    
        
    hist_MI_6mo, bin_edges_MI_6mo = np.histogram(MI_6mo_arr,nbins_MI_6mo)
    
    plt.figure()
    plt.bar(bin_edges_MI_6mo[:-1],hist_MI_6mo,width=.2,color='green',edgecolor='black',align='edge')
    plt.xlabel('Mutual information')
    plt.ylabel('Count')
    plt.title(pt+' 6 month on target Mutual information surrogate data')       