#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 22:36:06 2018

@author: virati
Bring in and view tractography
Focus on differentiating ONTarget and OFFTarget

OBSOLETE
"""

import numpy as np
import nibabel
import nilearn
from nilearn import plotting, image
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn import surface

from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/')
import DBSpace as dbo
from DBSpace import nestdict

import DBSpace.control.DTI as DTI

Etrode_map = DTI.Etrode_map

all_pts = ['906','907','908']
voltage = '2'

dti_file = nestdict()
data = nestdict()
data_arr = np.zeros((6,2,2,182,218,182))
combined = nestdict()

#fsaverage = datasets.fetch_surf_fsaverage5()


for pp,pt in enumerate(all_pts):
    for cc,condit in enumerate(['OnT','OffT']):
        for ss,side in enumerate(['L','R']):
            cntct = Etrode_map[condit][pt][ss]+1
            dti_file[pp][condit][side] = '/home/virati/Dropbox/projects/Research/MDD-DBS/Data/Anatomy/DTI/MDT_DBS_2_7V_Tractography/DBS'+str(pt) + '.'+side+str(cntct)+'.' + voltage + 'V.bin.nii.gz'
        
            data[pp][condit][side] = image.smooth_img(dti_file[pp][condit][side],fwhm=1)
            #plotting.plot_img(data[pp])
            
            #texture = surface.vol_to_surf(data[pp][condit][side],fsaverage.pial_right)
            #plotting.plot_surf_stat_map(fsaverage.infl_right,texture,hemi='right',title='Surf',threshold=1,bg_map=fsaverage.sulc_right,cmap='cold_hot')
            
            data_arr[pp,cc,ss,:,:,:] = np.array(data[pp][condit][side].dataobj)
                
        #combined = (data_arr[pp]['L'] + data_arr[pp]['R'])/2.0
        #combined[pt][condit] = nilearn.image.math_img("img1 + img2",img1=data[pp][condit]['L'],img2=data[pp][condit]['R'])
        #
        combined[pt][condit] = image.math_img("img1+img2",img1=data[pp][condit]['L'],img2=data[pp][condit]['R'])
        #locs = np.where(np.array(combined != 0))
        #fig = plt.figure()
        #plt.suptitle(pt)
        #ax = fig.add_subplot(111,projection='3d')
        #rndss = np.round(locs[0].shape[0]* np.random.uniform(0,1,100)).astype(np.int)
        #ax.scatter(locs[0][rndss],locs[1][rndss],locs[2][rndss],s=100)
    
        #plotting.plot_glass_brain(data[pp]['L'],black_bg=True)
        
        #plotting.plot_glass_brain(nibabel.Nifti1Image(data_arr[pp][condit]['L'],affine=-1*np.eye(4)),black_bg=True)
        
        #plotting.plot_glass_brain(combined[pt][condit],black_bg=True,title=pt + ' ' + condit)

#%%
#Find the mean for a condit
condit_avg = nestdict()
for cc, condit in enumerate(['OnT','OffT']):
    #condit_avg[condit] = image.math_img("np.mean(np.array([img4,img5,img6]),axis=0)",img1=combined['901'][condit],img2=combined['903'][condit],img3=combined['905'][condit],img4=combined['906'][condit],img5=combined['907'][condit],img6=combined['908'][condit])
    condit_avg[condit] = image.math_img("np.mean(np.array([img4,img5,img6]),axis=0)",img1=combined['901'][condit],img2=combined['903'][condit],img3=combined['905'][condit],img4=combined['906'][condit],img5=combined['907'][condit],img6=combined['908'][condit])
    #condit_avg[condit] = nilearn.image.math_img("img1+img2+img3+img4+img5+img6",img1=combined['901'][condit],img2=combined['903'][condit],img3=combined['905'][condit],img4=combined['906'][condit],img5=combined['907'][condit],img6=combined['908'][condit])
    plotting.plot_glass_brain(condit_avg[condit],black_bg=True,title=condit + ' average tractography',vmin=0,vmax=2)
    #plotting.plot_glass_brain(nibabel.Nifti1Image(np.median(np.sum(data_arr[:,0,:,:,:,:],axis=2),axis=0),affine=np.eye(4)))
#%%
diff_map = nestdict()
diff_map['OnT'] = image.math_img("(img1-img2) > 0.05",img1=condit_avg['OnT'],img2=condit_avg['OffT'])
diff_map['OffT'] = image.math_img("(img1-img2) < -0.05",img1=condit_avg['OnT'],img2=condit_avg['OffT'])

for target in ['OnT','OffT']:
    #plt.figure()
    #voxels = np.array(condit_avg[target].dataobj)
    #plt.hist(voxels.flatten(),bins=200,range=(0,1))
    
    plotting.plot_glass_brain(diff_map[target],black_bg=True,title=target + ' Preference Flow',vmin=-1,vmax=1)#,threshold=0.3)
    
    #test = np.mean(np.array(dti_file),axis=0)
    #plotting.plot_img(test)
    plt.show()
