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
from sklearn.utils import shuffle
from DBSpace.visualizations import EEG_Viz

from sklearn.model_selection import learning_curve

import seaborn as sns

sns.set_context('paper')
sns.set(font_scale=4)
sns.set_style('white')

#%%
regularization = 'l1'
do_null = False

#%%

# If we want to do channel masking
active_chann = [24,35,66,240,212]
limit_chann = False

#LOAD IN THE EEG FILE
#inFile = pickle.load(open('/home/virati/stream_intvs.pickle','rb'))
inFile = pickle.load(open('/home/virati/Dropbox/Data/streaming_EEG.pickle','rb'))

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
# Do the actualy train test split here
preshuff_ord = np.arange(0,labels.shape[0])
#Xtr,Xte,Ytr,Yte,buffnum_tr,buffnum_te,unshuff_ord_tr,unshuff_ord_te = sklearn.model_selection.train_test_split(dsgn_X,labels,times,preshuff_ord,test_size=0.33,shuffle=True)
Xtr,Xte,Ytr,Yte,unshuff_ord_tr,unshuff_ord_te = sklearn.model_selection.train_test_split(dsgn_X,labels,preshuff_ord,test_size=0.33,shuffle=True)

#NOW we do cross validation WITHIN the training set here

#%%
#Learning curve
tsize,tscore,vscore = learning_curve(svm.LinearSVC(penalty=regularization,dual=False,max_iter=10000),Xtr,Ytr,train_sizes=np.linspace(0.1,1,10),shuffle=True)

plt.figure()
plt.plot(tsize,np.mean(tscore,axis=1))
plt.plot(tsize,np.mean(vscore,axis=1))

#%%    
# Learn several models from train-test splitting
multi_accuracy = np.zeros((100,1))
multi_model = [None] * 100


# This needs to be converted to CV
for ii in range(10):
    #split our training set into 90% and 10% and do it 100 times
    X_cvtr,X_cvte,Y_cvtr,Y_cvte = sklearn.model_selection.train_test_split(Xtr,Ytr,test_size=0.10)

    clf = svm.LinearSVC(penalty=regularization,dual=False,max_iter=10000)
    
    # Do some null-testing through shuffling here
    if do_null:
        Y_cvtr = shuffle(Y_cvtr)
    
    # Fit the SVM
    clf.fit(X_cvtr,Y_cvtr)
    #cvtePreds = clf.predict(X_cvte)
    #multi_accuracy[ii] = sum(cvtePreds == Y_cvte) / len(cvtePreds)
    multi_accuracy[ii] = clf.score(X_cvte,Y_cvte)
    multi_model[ii] = clf
    
    
    
iimodel_max = np.argmax(multi_accuracy)


'''
This should all be made into a simple:
learn model
plot model

in a loop


'''

#%%
clf = multi_model[iimodel_max]

# If we want to limit the channels we're looking at
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
pickle.dump(clf,open('/tmp/Stream_SVMModel_'+regularization,'wb'))

#%%

## We'll plot the segments

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

# Plot the coefficients here
coeffs = clf.coef_.reshape(3,257,-1,order='C')

coeff_mag = [None] * 3
choose_band = 4
for stimclass in range(3):
    plt.figure()
    #plt.subplot(1,2,stimcla)
    plt.imshow(coeffs[stimclass,:,:])
    
    #coeff_mag[stimclass] = np.linalg.norm(coeffs[stimclass,:,:],axis=1)
    coeff_mag[stimclass] = coeffs[stimclass,:,choose_band]
    
    mainfig = plt.figure()
    EEG_Viz.plot_3d_scalp(coeff_mag[stimclass],mainfig,clims=(0,0.06),animate=False,label=label_map[stimclass],unwrap=True)
    plt.title(label_map[stimclass])
    
    
#%%
# need to do rPCA here for coefficients
    