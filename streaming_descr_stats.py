#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 12:48:04 2018

@author: virati
Streaming EEG descriptive statistics
"""


import sklearn
from sklearn import svm
import numpy as np
import pickle
import DBSpace as dbo

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from DBSpace.visualizations import EEG_Viz

from sklearn.model_selection import learning_curve

import seaborn as sns

from sklearn.decomposition import PCA, FastICA
import random
from scipy.stats import mode

sns.set_context('paper')
sns.set(font_scale=4)
sns.set_style('white')

#%%
regularization = 'l1'
do_null = False

#%%

#LOAD IN THE EEG FILE
#inFile = pickle.load(open('/home/virati/stream_intvs.pickle','rb'))
inFile = pickle.load(open('/home/virati/Dropbox/Data/streaming_EEG.pickle','rb'))

def do_ica():
    state_vec = np.array(inFile['States'][3][0]).reshape(-1,257,5,order='F') #ordering here as F results in curves that make more sense, namely that gamma is ALWAYS low
    state_vec = state_vec[:,:,2]

    ica = FastICA(n_components=6)
    S_ = ica.fit_transform(state_vec)
    A_ = ica.mixing_
    
    plt.plot(S_)
    
do_ica()

def do_descr():
    #example plot the power across all segments of a single condit
    state_vec = np.array(inFile['States'][3][0]).reshape(-1,257,5,order='F') #ordering here as F results in curves that make more sense, namely that gamma is ALWAYS low
    
    def jk_median(state_vecs):
        n_iter = 100
        med_vec = []
        #generate random list of integers between 0 and state_vec interval size
        for ii in range(n_iter):
            idxs = random.sample(range(0,state_vecs.shape[0]),30)
            med_vec.append(np.median(state_vecs[idxs,:,:],axis=0))
    
        return np.array(med_vec)
    ens_med = jk_median(state_vec)
    
    #%%
    plot_med = np.mean(ens_med,axis=0)
    for bb in range(5):
        mainfig = plt.figure()
        
        EEG_Viz.plot_3d_scalp(plot_med[:,bb],mainfig,clims=(-1,1),animate=False,unwrap=True)
        plt.title(dbo.feat_order[bb])