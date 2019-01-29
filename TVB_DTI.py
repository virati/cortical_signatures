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
parcel_coords = np.load('/home/virati/Dropbox/TVB_192_coord.npy')
# Load in a simple DTI image
condit = 'OnT'
pt = '906'
voltage = str(3)

#%% Load in the file

data = nestdict()
dti_file = nestdict()
for ss,side in enumerate(['L','R']):
    cntct = Etrode_map[condit][pt][ss]+1
    dti_file[side] = '/home/virati/Dropbox/projects/Research/MDD-DBS/Data/Anatomy/DTI/MDT_DBS_2_7V_Tractography/DBS'+str(pt) + '.'+side+str(cntct)+'.'+voltage+'V.bin.nii.gz'

    data[side] = image.smooth_img(dti_file[side],fwhm=1)

# Combine the images
combined = image.math_img("img1+img2",img1=data['L'],img2=data['R'])
#plotting.plot_glass_brain(combined,black_bg=True,title=pt + ' ' + condit)

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

dist_to_closest_tract = [None] * parcel_coords.shape[0]
for ii in range(parcel_coords.shape[0]):
    tract_dist = []
    for jj in range(display_vox_loc.shape[0]):
        tract_dist.append(np.linalg.norm(parcel_coords[ii,:] - display_vox_loc[jj,:]))
    
    dist_to_closest_tract[ii] = np.min(np.array(tract_dist))

dist_to_closest_tract = np.array(dist_to_closest_tract)



#%% Threshold tract -> parcellations
#This is our FIRST threshold
plt.hist(dist_to_closest_tract)
prior_locs = dist_to_closest_tract < 30

#%%
#So, above, we've just found the prior parcellations that we expect changes in
#No we're going to find the 2nd order nodes
first_order = prior_locs.astype(np.float)

f_laplacian = np.load('/home/virati/Dropbox/TVB_192_conn.npy')
second_order = np.dot(f_laplacian,first_order)
third_order = np.dot(f_laplacian,second_order)

def plot_first_scnd(first_order,second_order,fl):
    plt.figure()
    plt.subplot(221)
    plt.plot(first_order)
    plt.subplot(222)
    plt.plot(second_order)
    plt.subplot(2,1,2)
    plt.imshow(fl)

plot_first_scnd(first_order,second_order,f_laplacian)

second_locs = second_order > 20

#%%
#
eeg_scale = 10
EEG_coords = EEG_Viz.get_coords(scale=eeg_scale)

# Find First order EEG channels
dist_to_closest_parcel = [None] * EEG_coords.shape[0]
for cc in range(EEG_coords.shape[0]):
    parcel_dist = []
    for jj in parcel_coords[prior_locs]:
        parcel_dist.append(np.linalg.norm(EEG_coords[cc,:] - jj))
        
    dist_to_closest_parcel[cc] = np.min(np.array(parcel_dist))


# Find second order EEG channels
dist_to_closest_second = [None] * EEG_coords.shape[0]
for cc in range(EEG_coords.shape[0]):
    parcel_dist = []
    for jj in parcel_coords[second_locs]:
        parcel_dist.append(np.linalg.norm(EEG_coords[cc,:] - jj))
        
    dist_to_closest_second[cc] = np.min(np.array(parcel_dist))
    
#%%

eeg_thresh = 45
# This is our SECOND threshold
prior_channs = np.array(dist_to_closest_parcel) < eeg_thresh
plt.figure()
plt.hist(dist_to_closest_parcel)

second_channs = np.array(dist_to_closest_second) < eeg_thresh

chann_mask = np.zeros((257,))
chann_mask[prior_channs] = 1

second_chann_mask = np.zeros((257,))
second_chann_mask[second_channs] = 1

second_chann_mask = np.logical_and(second_chann_mask == 1, ~(chann_mask == 1)).astype(np.int)

#%%
#Plotting stuff now
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax._axis3don = False
ax.scatter(parcel_coords[:,0],parcel_coords[:,1],parcel_coords[:,2],s=200,alpha=0.5)
plt.title('Coordinates of the brain regions')


ax.scatter(display_vox_loc[:,0],display_vox_loc[:,1],display_vox_loc[:,2],alpha=0.7,s=100)
ax.scatter(parcel_coords[prior_locs,0],parcel_coords[prior_locs,1],parcel_coords[prior_locs,2],s=500,color='r')

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.grid(False)
#%%
#Now overlay the EEG channels
EEG_Viz.plot_3d_locs(np.ones((257,)),ax,scale=eeg_scale,animate=False)

EEG_Viz.plot_3d_scalp(chann_mask,ax,scale=10,alpha=0.5,unwrap=False)

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
EEG_Viz.plot_3d_scalp(second_chann_mask,ax,scale=10,alpha=0.2,unwrap=False)
plt.title('Secondary Channels')

#%%
fig = plt.figure()
ax = fig.add_subplot(111)
EEG_Viz.plot_3d_scalp(chann_mask,ax,scale=10,alpha=0.5,unwrap=True)
plt.title('Primary Channels')

fig = plt.figure()
ax = fig.add_subplot(111)
EEG_Viz.plot_3d_scalp(second_chann_mask,ax,scale=12,alpha=0.5,unwrap=True)
plt.title('Secondary Channels')


#%%
#Channel mask writing
EEG_support = {'primary':chann_mask,'secondary':second_chann_mask}
pickle.dump(EEG_support,open('/tmp/' + pt + '_' + condit + '_' + voltage,'wb'))


#%%
#Do Mayavi Plotting
#EEG_Viz.plot_maya_scalp(chann_mask,scale=10,alpha=0.5,unwrap=False)
#EEG_Viz.plot_maya_scalp(np.ones((257,)),ax,scale=eeg_scale,animate=False)
#EEG_Viz.plot_maya_scalp(chann_mask,ax,scale=10,alpha=0.5,unwrap=False)


EEG_Viz.plot_tracts(display_vox_loc,active_mask=[True]*display_vox_loc.shape[0],color=(1.,0.,0.))
EEG_Viz.plot_maya(display_vox_loc,active_mask=[True]*display_vox_loc.shape[0],color=(1.,0.,0.))
EEG_Viz.plot_maya(parcel_coords,active_mask=prior_locs,color=(0.,1.,0.))
EEG_Viz.plot_maya(parcel_coords,active_mask=second_locs,color=(0.,1.,1.))
EEG_Viz.plot_maya_scalp(chann_mask,ax,scale=10,alpha=0.5,unwrap=False)
EEG_Viz.plot_maya_scalp(second_chann_mask,ax,color=(0.,0.,1.),scale=10,alpha=0.3,unwrap=False)

#%%
plt.figure()
plt.hist(dist_to_closest_tract)