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

import mayavi.mlab as mlab
from mayavi.mlab import *

import pdb

#%%

class head_model:
    def __init__(self):
        self.electrode_map = DTI.Etrode_map
        self.parcel_coords = []
        self.dti_coords = []
        self.eeg_coords = []
        
    def import_parcellation(self):
        self.parcel_coords = 0.8 * np.load('/home/virati/Dropbox/TVB_192_coord.npy')
        
    def import_dti(self):    
        data = nestdict()
        dti_file = nestdict()
        for ss,side in enumerate(['L','R']):
            cntct = self.electrode_map[condit][pt][ss]+1
            dti_file[side] = '/home/virati/Dropbox/projects/Research/MDD-DBS/Data/Anatomy/DTI/MDT_DBS_2_7V_Tractography/DBS'+str(pt) + '.'+side+str(cntct)+'.'+str(voltage)+'V.bin.nii.gz'
        
            data[side] = image.smooth_img(dti_file[side],fwhm=1)
        
        # Combine the images
        combined = image.math_img("img1+img2",img1=data['L'],img2=data['R'])
        #plotting.plot_glass_brain(combined,black_bg=True,title=pt + ' ' + condit)
        
        #%% DTI STUFF
        #manual, numpy way
        voxels = np.array(combined.dataobj)
        vox_loc = np.array(np.where(voxels > 0)).T
        
        mni_vox = []
        for vv in vox_loc:
            mni_vox.append(np.array(image.coord_transform(vv[0],vv[1],vv[2],niimg.affine)))
        
        mni_vox = sig.detrend(np.array(mni_vox),axis=0,type='constant')
        
        #%% CALCULATE TRACT->PARCEL
        # now that we're coregistered, we go to each parcellation and find the minimum distance from it to the tractography
        
        vox_loc = mni_vox / 3
        
        display_vox_loc = vox_loc[np.random.randint(vox_loc.shape[0],size=(1000,)),:] / 3
        display_vox_loc += np.random.normal(0,1,size=display_vox_loc.shape)
        z_translate = np.zeros_like(display_vox_loc)
        z_translate[:,2] = 1
        y_translate = np.zeros_like(display_vox_loc)
        y_translate[:,1] = 1
        tract_offset = 15
        self.dti_coords += brain_offset * z_translate + tract_offset * y_translate
    
    def import_coords(self):
        pass
    
    def primary_nodes(self):
        pass
    
    def secondary_nodes(self):
        pass
    
    def viz_head(self):

        mlab.figure(bgcolor=(1.0,1.0,1.0))
        ## NEED TO PRETTY THIS UP with plot_3d_scalp updates that give much prettier OnT/OffT pictures
        # First, we plot the tracts from the DTI
        EEG_Viz.plot_tracts(self.dti_coords,active_mask=[True]*self.dti_coords.shape[0],color=(1.,0.,0.))
        EEG_Viz.plot_coords(self.dti_coords,active_mask=[True]*self.dti_coords.shape[0],color=(1.,0.,0.))
        
        # Next, we plot the parcellation nodes from TVB
        EEG_Viz.plot_coords(self.parcel_coords,active_mask=prior_locs,color=(0.,1.,0.))
        EEG_Viz.plot_coords(self.parcel_coords,active_mask=second_locs,color=(0.,1.,1.))
        
        # Finally, we plot the EEG channels with their primary and secondary masks
        EEG_Viz.plot_maya_scalp(chann_mask,ax,scale=10,alpha=0.5,unwrap=False)
        EEG_Viz.plot_maya_scalp(second_chann_mask,ax,color=(0.,0.,1.),scale=10,alpha=0.3,unwrap=False)
        
    
    def plot_mechanism(self):
        #%% Here we plot for the primary and secondary channels
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        EEG_Viz.plot_3d_scalp(second_chann_mask,ax,scale=10,alpha=0.2,unwrap=False)
        plt.title('Secondary Channels')
    
        fig = plt.figure()
        ax = fig.add_subplot(111)
        EEG_Viz.plot_3d_scalp(chann_mask,ax,scale=10,alpha=0.5,unwrap=True)
        plt.title('Primary Channels')
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        EEG_Viz.plot_3d_scalp(second_chann_mask,ax,scale=12,alpha=0.5,unwrap=True)
        plt.title('Secondary Channels')

def DTI_support_model(pt,voltage,dti_parcel_thresh=70,eeg_thresh=70,condit='OnT'):
    Etrode_map = DTI.Etrode_map
    # Load in the coordinates for the parcellation
    parcel_coords = 0.84 * np.load('/home/virati/Dropbox/TVB_192_coord.npy')
    # Load in the DTI coordinates
    brain_offset = 25
    tract_offset = 50 #this gives us forward/backward offset?
    dti_scale_factor = 0.8
    
    
    data = nestdict()
    dti_file = nestdict()
    for ss,side in enumerate(['L','R']):
        cntct = Etrode_map[condit][pt][ss]+1
        dti_file[side] = '/home/virati/Dropbox/projects/Research/MDD-DBS/Data/Anatomy/DTI/MDT_DBS_2_7V_Tractography/DBS'+str(pt) + '.'+side+str(cntct)+'.'+str(voltage)+'V.bin.nii.gz'
    
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
    vox_loc = mni_vox/dti_scale_factor #This scale factor is for the tractography. We want to make sure the tracts, especially the dorsal aspect of the cingulum, makes sense wrt the location of the rest of it

    #%%
    # now that we're coregistered, we go to each parcellation and find the minimum distance from it to the tractography
    display_vox_loc = vox_loc[np.random.randint(vox_loc.shape[0],size=(1000,)),:] / 3
    display_vox_loc += np.random.normal(0,1,size=display_vox_loc.shape)
    z_translate = np.zeros_like(display_vox_loc)
    z_translate[:,2] = 1
    y_translate = np.zeros_like(display_vox_loc)
    y_translate[:,1] = 1
    display_vox_loc += brain_offset * z_translate + tract_offset * y_translate
    
    
    dist_to_closest_tract = [None] * parcel_coords.shape[0]
    for ii in range(parcel_coords.shape[0]):
        tract_dist = []
        for jj in range(display_vox_loc.shape[0]):
            tract_dist.append(np.linalg.norm(parcel_coords[ii,:] - display_vox_loc[jj,:]))
        
        dist_to_closest_tract[ii] = np.min(np.array(tract_dist))
    
    dist_to_closest_tract = np.array(dist_to_closest_tract)
    
    
    
    #%% Threshold tract -> parcellations
    #This is our FIRST threshold
    

    
    #plt.hist(dist_to_closest_tract)
    prior_locs = dist_to_closest_tract < dti_parcel_thresh
    plt.figure();plt.hist(dist_to_closest_tract)
    plt.title('Tract->Parcel histogram')
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
    
    #plot_first_scnd(first_order,second_order,f_laplacian)
    #This value is chosen semi-randomly to achieve a reasonable secondary-EEG density
    
    plt.figure();plt.hist(second_order)
    plt.title('Histogram of Second Order Laplacian Magnitudes')
    second_locs = second_order > 1
    #%%
    #
    eeg_scale = 10
    EEG_coords = EEG_Viz.get_coords(scale=eeg_scale)
    # maybe scale things here..
    
    # Find First order EEG channels
    dist_to_closest_parcel = [None] * EEG_coords.shape[0]
    for cc in range(EEG_coords.shape[0]):
        parcel_dist = []
        for jj in parcel_coords[prior_locs]:
            parcel_dist.append(np.linalg.norm(EEG_coords[cc,:] - jj))
            
        dist_to_closest_parcel[cc] = np.min(np.array(parcel_dist))
    
    #pdb.set_trace()
    # Find second order EEG channels
    dist_to_closest_second = [None] * EEG_coords.shape[0]
    for cc in range(EEG_coords.shape[0]):
        parcel_dist = []
        for jj in parcel_coords[second_locs]:
            parcel_dist.append(np.linalg.norm(EEG_coords[cc,:] - jj))
            
        dist_to_closest_second[cc] = np.min(np.array(parcel_dist))
        
    #%%
    
    # This is our SECOND threshold
    prior_channs = np.array(dist_to_closest_parcel) < eeg_thresh
    
    #plt.figure()
    #plt.hist(dist_to_closest_parcel)
    
    second_channs = np.array(dist_to_closest_second) < eeg_thresh
    
    chann_mask = np.zeros((257,))
    chann_mask[prior_channs] = 1
    
    second_chann_mask = np.zeros((257,))
    second_chann_mask[second_channs] = 1
    
    second_chann_mask = np.logical_and(second_chann_mask == 1, ~(chann_mask == 1)).astype(np.int)
    
    #pdb.set_trace()
    #%%
    #Channel mask writing
    EEG_support = {'primary':chann_mask,'secondary':second_chann_mask,
                   'parcel_coords':parcel_coords,'prior_locs':prior_locs,'eeg_scale':eeg_scale,
                   'second_locs':second_locs,'dti_scale_factor':dti_scale_factor,
                   'brain_offset':brain_offset,'tract_offset':tract_offset}
    #pickle.dump(EEG_support,open('/tmp/' + pt + '_' + condit + '_' + voltage,'wb'))
    return EEG_support
    
    
    #%%
    #Do Mayavi Plotting
    #EEG_Viz.plot_maya_scalp(chann_mask,scale=10,alpha=0.5,unwrap=False)
    #EEG_Viz.plot_maya_scalp(np.ones((257,)),ax,scale=eeg_scale,animate=False)
    #EEG_Viz.plot_maya_scalp(chann_mask,ax,scale=10,alpha=0.5,unwrap=False)


#%%
    # The main support model code
def plot_support_model(EEG_support,pt,voltage=3,condit='OnT'):
    Etrode_map = DTI.Etrode_map
    brain_offset = EEG_support['brain_offset']
    dti_scale_factor = EEG_support['dti_scale_factor']
    tract_offset = EEG_support['tract_offset']
    #%%
    # Load in the coordinates for the parcellation
    second_locs = EEG_support['second_locs']
    
    #%%
    data = nestdict()
    dti_file = nestdict()
    for ss,side in enumerate(['L','R']):
        cntct = Etrode_map[condit][pt][ss]+1
        dti_file[side] = '/home/virati/Dropbox/projects/Research/MDD-DBS/Data/Anatomy/DTI/MDT_DBS_2_7V_Tractography/DBS'+str(pt) + '.'+side+str(cntct)+'.'+str(voltage)+'V.bin.nii.gz'
    
        data[side] = image.smooth_img(dti_file[side],fwhm=1)
    
    # Combine the images
    combined = image.math_img("img1+img2",img1=data['L'],img2=data['R'])
    #plotting.plot_glass_brain(combined,black_bg=True,title=pt + ' ' + condit)
    
    #%% DTI STUFF
    #manual, numpy way
    voxels = np.array(combined.dataobj)
    vox_loc = np.array(np.where(voxels > 0)).T
    
    
    parcel_coords = EEG_support['parcel_coords']
    prior_locs= EEG_support['prior_locs']
    eeg_scale = EEG_support['eeg_scale']
    
    
    
    #pdb.set_trace()
    z_translate = np.zeros_like(parcel_coords);
    z_translate[:,2] = 1
    parcel_coords = parcel_coords * [1,1.5,1] + brain_offset*z_translate
    

    
    mni_vox = []
    for vv in vox_loc:
        mni_vox.append(np.array(image.coord_transform(vv[0],vv[1],vv[2],niimg.affine)))
    
    mni_vox = sig.detrend(np.array(mni_vox),axis=0,type='constant')
    
    #%% CALCULATE TRACT->PARCEL
    # now that we're coregistered, we go to each parcellation and find the minimum distance from it to the tractography
    
    vox_loc = mni_vox / dti_scale_factor
    
    display_vox_loc = vox_loc[np.random.randint(vox_loc.shape[0],size=(1000,)),:] / 3
    display_vox_loc += np.random.normal(0,1,size=display_vox_loc.shape)
    z_translate = np.zeros_like(display_vox_loc)
    z_translate[:,2] = 1
    y_translate = np.zeros_like(display_vox_loc)
    y_translate[:,1] = 1
    display_vox_loc += brain_offset * z_translate + tract_offset * y_translate
    
    
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
#    
#    
#    #%% Now we move to the EEG scaling
#    #Now overlay the EEG channels
    EEG_Viz.plot_3d_locs(np.ones((257,)),ax,scale=eeg_scale,animate=False)
    chann_mask = EEG_support['primary']
    second_chann_mask = EEG_support['secondary']
    #EEG_Viz.plot_3d_scalp(chann_mask,ax,scale=eeg_scale,alpha=0.5,unwrap=False)
    
    
    #%% Here we plot for the primary and secondary channels
#    fig = plt.figure()
#    ax = fig.add_subplot(111,projection='3d')
#    EEG_Viz.plot_3d_scalp(second_chann_mask,ax,scale=10,alpha=0.2,unwrap=False)
#    plt.title('Secondary Channels')
#
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    EEG_Viz.plot_3d_scalp(chann_mask,ax,scale=10,alpha=0.5,unwrap=True)
#    plt.title('Primary Channels')
#    
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    EEG_Viz.plot_3d_scalp(second_chann_mask,ax,scale=12,alpha=0.5,unwrap=True)
#    plt.title('Secondary Channels')
    
    #%%
    EEG_Viz.maya_band_display(1*chann_mask - second_chann_mask)
    #%EEG_Viz.maya_band_display(-1*second_chann_mask)
    
    #%% FINAL PLOTTING
    mlab.figure(bgcolor=(1.0,1.0,1.0))
    ## NEED TO PRETTY THIS UP with plot_3d_scalp updates that give much prettier OnT/OffT pictures
    # First, we plot the tracts from the DTI
    EEG_Viz.plot_tracts(display_vox_loc,active_mask=[True]*display_vox_loc.shape[0],color=(1.,0.,0.))
    #EEG_Viz.plot_coords(display_vox_loc,active_mask=[True]*display_vox_loc.shape[0],color=(1.,0.,0.))
    
    # Next, we plot the parcellation nodes from TVB
    EEG_Viz.plot_coords(parcel_coords,active_mask=prior_locs,color=(0.,1.,0.))
    EEG_Viz.plot_coords(parcel_coords,active_mask=second_locs,color=(0.,0.,1.))
    
    # Finally, we plot the EEG channels with their primary and secondary masks
    EEG_Viz.plot_maya_scalp(chann_mask,color=(0.,1.,0.),scale=10,alpha=0.5,unwrap=False)
    EEG_Viz.plot_maya_scalp(second_chann_mask,color=(0.,0.,1.),scale=10,alpha=0.3,unwrap=False)
    
    #%%
#    plt.figure()
#    plt.hist(dist_to_closest_tract)
    
    

if __name__=='__main__':
    for pt in ['908']:
        for voltage in [4]:
            supp_model = DTI_support_model(pt,str(voltage),dti_parcel_thresh=20,eeg_thresh=50)
            plot_support_model(supp_model,pt=pt)