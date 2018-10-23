#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 16:38:20 2018

@author: virati
UMAP fun with coherences
"""


#import sys
#sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/')
import DBSpace as dbo
from DBSpace import nestdict

import matplotlib.pyplot as plt
import pickle
import numpy as np
import cmocean

from sklearn.decomposition import PCA, SparsePCA
import string

import umap
import pdb



pts = ['906','907']
on_label = {'OnT':'BONT','OffT':'BOFT'}

csd_dict = nestdict()
plv_dict = nestdict()


for pt in pts:
    with open('/home/virati/Dropbox/Data/DBS'+pt+'_coh_dict.pickle','rb') as handle:
        import_dict = pickle.load(handle)
        
        csd_dict[pt] = import_dict['CSD'][pt]
        plv_dict[pt] = import_dict['PLV'][pt]
        
#%%
for pt in pts:
    for condit in ['OnT','OffT']:
        for epoch in ['Off_3',on_label[condit]]:
            matrix = np.zeros((257,257))
            for ii in range(257):
                for jj in range(257):
                    matrix[ii,jj] = plv_dict[pt][condit][epoch][ii][jj]
                    
            plv_dict[pt][condit][epoch] = matrix
        
#%%
            
data = np.array([[[plv_dict[pt][condit][epoch] for epoch in ['Off_3',on_label[condit]]] for condit in ['OnT','OffT']] for pt in pts]).reshape(-1,257,257)

#%%
udim = umap.UMAP().fit_transform(data)
plt.figure()
plt.scatter(udim[:,0],udim[:,1])