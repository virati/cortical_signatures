#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 22:17:00 2018

@author: virati
PURE streaming classifier SVM
Trained and tested on the streamed, locally reference EEG data
This is the SECOND step in the streaming/online EEG pipeline

YOU HAVE TO FIRST RUN dEEG_streaming

"""


import sklearn
from sklearn import svm
import numpy as np
import pickle

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from EEG_Viz import plot_3d_scalp

from sklearn.model_selection import learning_curve

import seaborn as sns

sns.set_context('paper')
sns.set(font_scale=4)
sns.set_style('white')

active_chann = [24,35,66,240,212]
limit_chann = False

#LOAD IN THE EEG FILE
#inFile = pickle.load(open('/home/virati/stream_intvs.pickle','rb'))
inFile = pickle.load(open('/tmp/big_file.pickle','rb'))

#IF WE ADD LFP HERE WE'RE AWESOME



rec = inFile['States']
lab = inFile['Labels']
times = np.concatenate(np.array(inFile['Times']).reshape(-1))

dsgn_X = np.array(np.concatenate([np.concatenate([rec[pt][cdt] for cdt in range(2)]) for pt in range(3)]))
labels = np.array(np.concatenate([np.concatenate([lab[pt][cdt] for cdt in range(2)]) for pt in range(3)]))
raw_labels = np.copy(labels)
#Transform the labels to actual labels
label_map = {0:'OFF',2:'OnTON',1:'OffTON'}

labels = np.array([label_map[item] for item in labels])

#%%

#do iteration of model a few times
#clf = svm.LinearSVC(penalty='l2',dual=False)

#%%
if 0:
    plt.figure()
    #plt.plot(labels)
    plt.imshow(raw_labels.reshape(1,-1),aspect='auto')
    #just for display
    ddXtr,ddXte,ddYtr,ddYte,ddbuffnum_tr,ddbuffnum_te = sklearn.model_selection.train_test_split(dsgn_X,raw_labels,times,test_size=0.33)
    plt.figure()
    plt.imshow(np.hstack((ddYtr,ddYte)).reshape(1,-1),aspect='auto')
    plt.suptitle('Shuffled segments - JUST FOR DISPLAY')

#%%
preshuff_ord = np.arange(0,labels.shape[0])
#Xtr,Xte,Ytr,Yte,buffnum_tr,buffnum_te,unshuff_ord_tr,unshuff_ord_te = sklearn.model_selection.train_test_split(dsgn_X,labels,times,preshuff_ord,test_size=0.33,shuffle=True)
Xtr,Xte,Ytr,Yte,unshuff_ord_tr,unshuff_ord_te = sklearn.model_selection.train_test_split(dsgn_X,labels,preshuff_ord,test_size=0.33,shuffle=True)

#NOW we do cross validation WITHIN the training set here

#%%
#Learning curve
tsize,tscore,vscore = learning_curve(svm.LinearSVC(penalty='l2',dual=False),Xtr,Ytr,train_sizes=np.linspace(0.05,1,20),shuffle=True)

#%%
plt.figure()
plt.plot(tsize,np.mean(tscore,axis=1))
plt.plot(tsize,np.mean(vscore,axis=1))

#%%    
multi_accuracy = np.zeros((100,1))
multi_model = [None] * 100



for ii in range(1):
    #split our training set into 90% and 10% and do it 100 times
    print(ii)
    X_cvtr,X_cvte,Y_cvtr,Y_cvte = sklearn.model_selection.train_test_split(Xtr,Ytr,test_size=0.10)

    clf = svm.LinearSVC(penalty='l2',dual=False)
    clf.fit(X_cvtr,Y_cvtr)
    #cvtePreds = clf.predict(X_cvte)
    #multi_accuracy[ii] = sum(cvtePreds == Y_cvte) / len(cvtePreds)
    multi_accuracy[ii] = clf.score(X_cvte,Y_cvte)
    multi_model[ii] = clf
    
    
    
iimodel_max = np.argmax(multi_accuracy)


#%%
clf = multi_model[iimodel_max]


if limit_chann:
    #NOW let's do an artificial MASK!!! coup de grace
    fold_dsgn_X = Xte.reshape(-1,257,5,order='C')
    sub_X = np.zeros_like(fold_dsgn_X)
    
    sub_X[:,active_chann] = fold_dsgn_X[:,active_chann]
    
    sub_X = sub_X.reshape(-1,257*5)
    Xte_subset = sub_X

    preds = clf.predict(Xte_subset)
else:    
    preds = clf.predict(Xte)

Yte_labels = np.copy(Yte)
correct = sum(preds == Yte_labels)
accuracy = correct / len(preds)

nclass0 = sum(Yte_labels == 'OFF')
nclass1 = sum(Yte_labels == 'OnTON')
nclass2 = sum(Yte_labels == 'OffTON')
total = len(Yte)

print(accuracy)

#%%
#Analyse the error segments heres
#wrong_buffers = np.where(preds != Yte)
#plt.figure()
#plt.hist(buffnum_tr[wrong_buffers[0]],bins=20)

#%%

#Save the classifier here
pickle.dump(clf,open('/tmp/Stream_SVMModel_l2','wb'))

#%%

plt.figure()
plt.subplot(2,1,1)

Yte[Yte == 'OffTON'] = 2
Yte[Yte == 'OnTON'] = 1
Yte[Yte == 'OFF'] = 0

preds[preds== 'OffTON'] = 2
preds[preds== 'OnTON'] = 1
preds[preds== 'OFF'] = 0


plt.imshow(np.vstack((Yte.astype(np.int),preds.astype(np.int))),aspect='auto')
plt.title('Stacked shuffled')
plt.legend()
#plt.plot(preds,label='Prediction')
#plt.plot(Yte,label='Actual')


plt.subplot(2,1,2)
#plt.stem((preds == Yte).astype(np.int),label='HITS')
#unshuffle the testing dataset
buffnum_te = unshuff_ord_te
unshuffYte = np.zeros(labels.shape[0])
unshuffPreds = np.zeros(labels.shape[0])
unshuffYte[buffnum_te] = Yte
unshuffPreds[buffnum_te] = preds


plt.imshow(np.vstack((unshuffYte.astype(np.int),unshuffPreds.astype(np.int))),aspect='auto')
plt.title('Unshuffled Segments')
plt.legend()

conf_matrix = confusion_matrix(Yte,preds)
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
    plot_3d_scalp(coeff_mag[stimclass],mainfig,clims=(0,0.06),animate=False,label=label_map[stimclass],unwrap=True)
    plt.title(label_map[stimclass])