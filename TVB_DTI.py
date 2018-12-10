#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 11:37:00 2018

@author: virati
Main file to co-register the TVB coordinates and the tractography information
"""

import pickle
import numpy as np
import DTI
import DBSpace as dbo
from DBSpace import nestdict
import nilearn
from nilearn import plotting, image,datasets

niimg = datasets.load_mni152_template()



from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


#%%

Etrode_map = DTI.Etrode_map

# Load in the coordinates for the parcellation
coords = np.load('/tmp/192_coord.npy')
# Load in a simple DTI image
condit = 'OnT'
pt = '906'

data = nestdict()
dti_file = nestdict()
for ss,side in enumerate(['L','R']):
    cntct = Etrode_map[condit][pt][ss]+1
    dti_file[side] = '/home/virati/Dropbox/projects/Research/MDD-DBS/Data/Anatomy/DTI/MDT_DBS_2_7V_Tractography/DBS'+str(pt) + '.'+side+str(cntct)+'.4V.bin.nii.gz'

    data[side] = image.smooth_img(dti_file[side],fwhm=1)

        
combined = image.math_img("img1+img2",img1=data['L'],img2=data['R'])

#plotting.plot_glass_brain(combined,black_bg=True,title=pt + ' ' + condit)


# convert to numpy array?
threshold = image.threshold_img(combined,0)

#%%
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(coords[:,0],coords[:,1],coords[:,2])


#%%
#manual, numpy way
vox_loc = np.array(np.where(voxels > 0)).T
#randomly subsample just for shits and gigs and to actually run properly on display
display_vox_loc = vox_loc[np.random.randint(vox_loc.shape[0],size=(100,)),:]

fig2 = plt.figure()
ax = fig2.add_subplot(111,projection='3d')
ax.scatter(display_vox_loc[:,0],display_vox_loc[:,1],display_vox_loc[:,2],alpha=0.5,s=100)