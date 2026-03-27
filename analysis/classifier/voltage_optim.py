#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 21:24:09 2018

@author: virati
Voltage optimization using targeting classifier
"""



#import sys
#sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/')
import DBSpace as dbo
from DBSpace import nestdict
from DBSpace.control.stream_dEEG import streamEEG

import itertools
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import numpy as np

import pickle

perf_dict = nestdict()
import seaborn as sns


pts = ['906']
condits =  ['Volt']
class_type = 'l1'
#%%
pt_test = [None] * len(pts)
pt_test_labels = [None] * len(pts)

for pp,pt in enumerate(pts):
    pt_test[pp] = []
    pt_test_labels[pp] = []
    
    for condit in condits:
        print('Doing ' + pt + ' ' + condit)
        sEEG = streamEEG(ds_fact=2,pt=pt,condit=condit,spotcheck=True)
        sEEG.plot_TF(chann=32)
        #%%
        
        sEEG.seg_PSDs()
        
        #%%
        sEEG.calc_baseline()
        #%%
        sEEG.label_segments()
        if 1:
            #This CLASSIFIES the data
            perf_dict[pt][condit] = sEEG.classify_segs(ctype=class_type,train_type='stream')
        else:
            #This WRITES the data out:
            pt_test[pp].append(sEEG.gen_test_matrix())
            pt_test_labels[pp].append(sEEG.true_labels)
            
#%%
with open('/tmp/big_file.pickle','wb') as file:
    pickle.dump({'States':pt_test,'Labels':pt_test_labels},file)
    

#%%

results_matrix = [np.array((perf_dict[pt][condit][0],perf_dict[pt][condit][1])) for condit,pt in itertools.product(condits,pts)]
results_matrix = np.concatenate(results_matrix,axis=1)

#index 0 is pred, 1 is true

conf_matrix = confusion_matrix(results_matrix[1,:],results_matrix[0,:])

plt.figure()
plt.subplot(1,2,1)
plt.plot(results_matrix[1,:],label='True')
plt.plot(results_matrix[0,:],label='Predicted')

plt.subplot(1,2,2)
plt.imshow(conf_matrix)
plt.yticks(np.arange(0,3),['OFF','OffT','OnT'])
plt.xticks(np.arange(0,3),['OFF','OffT','OnT'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.colorbar()

np.save('/home/virati/pre_conf_matrix_'+class_type,results_matrix)



#%%
#Show the SVM's coefficients
coeffs = sEEG.clf.coef_
plt.figure()
plt.plot()



#%%
#sEEG.re_ref(scheme='local')

#%%
#DO STREAMING, SEGMENTED Osc Band Calculations

#sEEG.re_ref()
#%%
#sEEG.SG_Transform(nperseg=2**11,noverlap=2**11-50,ctype='virtual')
#sEEG.SG_Transform(nperseg=2**11,noverlap=2**11-50,ctype='real')

        

