#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 22:17:00 2018

@author: virati
PURE streaming classifier SVM
Trained and tested on the streamed, locally reference EEG data
"""


import sklearn
from sklearn import svm
import numpy as np
import pickle

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from EEG_Viz import plot_3d_scalp



active_chann = [24,35,66,240,212]
limit_chann = True

#LOAD IN THE EEG FILE
inFile = pickle.load(open('/home/virati/stream_intvs.pickle','rb'))

#IF WE ADD LFP HERE WE'RE AWESOME



rec = inFile['States']
lab = inFile['Labels']

dsgn_X = np.array(np.concatenate([np.concatenate([rec[pt][cdt] for cdt in range(2)]) for pt in range(3)]))
labels = np.array(np.concatenate([np.concatenate([lab[pt][cdt] for cdt in range(2)]) for pt in range(3)]))
#Transform the labels to actual labels
label_map = {0:'OFF',2:'OnTON',1:'OffTON'}

labels = np.array([label_map[item] for item in labels])

#%%

#do iteration of model a few times
for ii in range(1):
    clf = svm.LinearSVC(penalty='l2',dual=False)
    
    Xtr,Xte,Ytr,Yte = sklearn.model_selection.train_test_split(dsgn_X,labels,test_size=0.33)
    
    clf.fit(Xtr,Ytr)
    
    
    if limit_chann:
        #NOW let's do an artificial MASK!!! coup de grace
        fold_dsgn_X = Xte.reshape(-1,257,5,order='C')
        sub_X = np.zeros_like(fold_dsgn_X)
        
        sub_X[:,active_chann] = fold_dsgn_X[:,active_chann]
        
        sub_X = sub_X.reshape(-1,257*5)
        Xte_subset = sub_X
    
        ##
    
        preds = clf.predict(Xte_subset)
    else:    
        preds = clf.predict(Xte)

    correct = sum(preds == Yte)
    accuracy = correct / len(preds)
    
    nclass0 = sum(Yte == 'OFF')
    nclass1 = sum(Yte == 'OnTON')
    nclass2 = sum(Yte == 'OffTON')
    total = len(Yte)
    
    print(accuracy)

#%%

#Save the classifier here
pickle.dump(clf,open('/tmp/Stream_SVMModel_l2','wb'))

#%%

plt.figure()
plt.subplot(2,1,1)
plt.plot(preds,label='Prediction')
plt.plot(Yte,label='Actual')
plt.legend()
plt.subplot(2,1,2)
plt.stem((preds == Yte).astype(np.int),label='HITS')
plt.legend()

conf_matrix = confusion_matrix(preds,Yte)
plt.figure()
plt.imshow(conf_matrix)
plt.yticks(np.arange(0,3),['OFF','OffT','OnT'])
plt.xticks(np.arange(0,3),['OFF','OffT','OnT'])
plt.colorbar()


#%%
coeffs = clf.coef_.reshape(3,257,-1,order='C')

coeff_mag = [None] * 3
for stimclass in range(3):
    plt.figure()
    #plt.subplot(1,2,stimcla)
    plt.imshow(coeffs[stimclass,:,:])
    
    coeff_mag[stimclass] = np.linalg.norm(coeffs[stimclass,:,:],axis=1)
    
    mainfig = plt.figure()
    plot_3d_scalp(coeff_mag[stimclass],mainfig,clims=(0,0.06))
    plt.title(label_map[stimclass])