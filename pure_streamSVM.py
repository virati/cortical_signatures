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



#LOAD IN THE EEG FILE
inFile = pickle.load(open('/home/virati/stream_intvs.pickle','rb'))

#IF WE ADD LFP HERE WE'RE AWESOME



rec = inFile['States']
lab = inFile['Labels']

dsgn_X = np.array(np.concatenate([np.concatenate([rec[pt][cdt] for cdt in range(2)]) for pt in range(3)]))
labels = np.array(np.concatenate([np.concatenate([lab[pt][cdt] for cdt in range(2)]) for pt in range(3)]))

clf = svm.LinearSVC(penalty='l2',dual=False)
Xtr,Xte,Ytr,Yte = sklearn.model_selection.train_test_split(dsgn_X,labels,test_size=0.33)

clf.fit(Xtr,Ytr)

preds = clf.predict(Xte)


#%%

plt.figure()
plt.plot(preds)
plt.plot(Yte)

conf_matrix = confusion_matrix(preds,Yte)
plt.figure()
plt.imshow(conf_matrix)
plt.yticks(np.arange(0,3),['OFF','OffT','OnT'])
plt.xticks(np.arange(0,3),['OFF','OffT','OnT'])
plt.colorbar()

correct = sum(preds == Yte)
accuracy = correct / len(preds)

nclass0 = sum(Yte == 0)
nclass1 = sum(Yte == 1)
nclass2 = sum(Yte == 2)
total = len(Yte)

print(accuracy)

#%%
coeffs = clf.coef_.reshape(3,257,-1,order='C')
plt.figure()
for stimclass in range(3):
    plt.subplot(1,3,stimclass+1)
    plt.imshow(coeffs[stimclass,:,:])