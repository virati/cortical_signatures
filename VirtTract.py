#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 13:23:17 2019

@author: virati
Class for DTI->TVB connection
"""

'''
Virtual Tractography Class
'''

class VT:
    def __init__(self,pt,condit='OnT',voltage=4,tvb_conn_map=192):
        self.Etrode_map = DTI.Etrode_map
        self.parcel_coords = np.load('/home/virati/Dropbox/TVB_' + tvb_conn_map + '_coord.npy')
        self.condit = condit
        self.pt = pt
        self.stim_voltage = voltage
        
    def load_DTI(self):
        
        data = nestdict()
        dti_file = nestdict()
        for ss,side in enumerate(['L','R']):
            cntct = Etrode_map[condit][pt][ss]+1
            dti_file[side] = '/home/virati/Dropbox/projects/Research/MDD-DBS/Data/Anatomy/DTI/MDT_DBS_2_7V_Tractography/DBS'+str(pt) + '.'+side+str(cntct)+'.'+voltage+'V.bin.nii.gz'
        
            data[side] = image.smooth_img(dti_file[side],fwhm=1)
        
        # Combine the images
        self.combined = image.math_img("img1+img2",img1=data['L'],img2=data['R'])
        
    def dti_to_mni(self):
        voxels = np.array(self.combined.dataobj)
        vox_loc = np.array(np.where(voxels > 0)).T
        
        mni_vox = []
        for vv in vox_loc:
            mni_vox.append(np.array(image.coord_transform(vv[0],vv[1],vv[2],niimg.affine)))
        
        self.mni_vox = sig.detrend(np.array(mni_vox),axis=0,type='constant')
        
    def dist_T_to_P(self,thresh=30):
        vox_loc = self.mni_vox
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
        #plt.hist(dist_to_closest_tract)
        self.primary_parcels = dist_to_closest_tract < 30
        
    def compute_2ndparcels(self):
        inputs_threshold = 20
        first_order = prior_locs.astype(np.float)

        f_laplacian = np.load('/home/virati/Dropbox/TVB_192_conn.npy')
        second_order = np.dot(f_laplacian,first_order)
        third_order = np.dot(f_laplacian,second_order)
        
        self.secondary_parcels = second_order > inputs_threshold
        
    def dist_P_to_EEG(self,threshold=45):
        pass


