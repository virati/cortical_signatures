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

import scipy.signal as sig
niimg = datasets.load_mni152_template()

import DBSpace
from DBSpace.visualizations import EEG_Viz


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
#Move now to the DTI stuff directly
#manual, numpy way
voxels = np.array(combined.dataobj)
vox_loc = np.array(np.where(voxels > 0)).T

mni_vox = []
for vv in vox_loc:
    mni_vox.append(np.array(image.coord_transform(vv[0],vv[1],vv[2],niimg.affine)))

mni_vox = sig.detrend(np.array(mni_vox),axis=0,type='constant')

#%%
# now that we're coregistered, we go to each parcellation and find the minimum distance from it to the tractography


vox_loc = mni_vox
#randomly subsample just for shits and gigs and to actually run properly on display
display_vox_loc = vox_loc[np.random.randint(vox_loc.shape[0],size=(1000,)),:] / 3
display_vox_loc += np.random.normal(0,1,size=display_vox_loc.shape)
#ax = fig.add_subplot(111,projection='3d')

dist_to_closest_tract = [None] * coords.shape[0]
for ii in range(coords.shape[0]):
    tract_dist = []
    for jj in range(display_vox_loc.shape[0]):
        tract_dist.append(np.linalg.norm(coords[ii,:] - display_vox_loc[jj,:]))
    
    dist_to_closest_tract[ii] = np.min(np.array(tract_dist))

dist_to_closest_tract = np.array(dist_to_closest_tract)
prior_locs = dist_to_closest_tract < 30

#%%
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax._axis3don = False
ax.scatter(coords[:,0],coords[:,1],coords[:,2],s=200,alpha=0.5)
plt.title('Coordinates of the brain regions')

ax.scatter(display_vox_loc[:,0],display_vox_loc[:,1],display_vox_loc[:,2],alpha=0.4,s=50)
ax.scatter(coords[prior_locs,0],coords[prior_locs,1],coords[prior_locs,2],s=500,color='r')
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.grid(False)

EEG_coords = EEG_Viz.get_coords()

dist_to_closest_parcel = [None] * EEG_coords.shape[0]
for cc in range(EEG_coords.shape[0]):
    parcel_dist = []
    for jj in coords[prior_locs]:
        parcel_dist.append(np.linalg.norm(EEG_coords[cc,:] - jj))
        
    dist_to_closest_parcel[cc] = np.min(np.array(parcel_dist))

prior_channs = np.array(dist_to_closest_parcel) < 10
#Now overlay the EEG channels
EEG_Viz.plot_3d_locs(np.ones((257,)),ax,scale=10,animate=False)

chann_mask = np.zeros((257,))
chann_mask[prior_channs] = 1

EEG_Viz.plot_3d_scalp(chann_mask,ax,scale=10)



#%%
plt.figure()
plt.hist(dist_to_closest_tract)