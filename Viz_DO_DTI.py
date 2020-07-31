#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 22:36:06 2018

@author: virati
Bring in and view tractography
Focus on differentiating ONTarget and OFFTarget

This one now looks at the DO tractography at granularity of left/right, Ont/OffT, patients, etc.
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
import itertools

Etrode_map = DTI.Etrode_map

all_pts = ['901','903','905','906','907','908']
all_condits = ['OnT','OffT']
all_sides = ['L','R','L+R']
voltage = '3'

DO_all = itertools.product(all_pts,all_condits,all_sides)
DO_positive = [('901','OnT','L'),('901','OffT','L'),('903','OffT','L'),('905','OnT','L+R'),('905','OffT','L+R'),('906','OffT','R')]
DO_negative = [x for x in DO_all if x not in DO_positive]

dti_file = nestdict()
data = nestdict()
tractos = nestdict()

data_arr = np.zeros((6,2,2,182,218,182))
combined = nestdict()

fsaverage = datasets.fetch_surf_fsaverage5()


for pp,pt in enumerate(all_pts):
    for cc,condit in enumerate(['OnT','OffT']):
        for ss,side in enumerate(['L','R']):
            cntct = Etrode_map[condit][pt][ss]+1
            dti_file[pp][condit][side] = '/home/virati/Dropbox/projects/Research/MDD-DBS/Data/Anatomy/DTI/MDT_DBS_2_7V_Tractography/DBS'+str(pt) + '.'+side+str(cntct)+'.' + voltage + 'V.bin.nii.gz'
        
            tractos[pt][condit][side] = image.smooth_img(dti_file[pp][condit][side],fwhm=1)

            data_arr[pp,cc,ss,:,:,:] = np.array(tractos[pt][condit][side].dataobj)
                

        tractos[pt][condit]['L+R'] = image.math_img("img1+img2",img1=tractos[pt][condit]['L'],img2=tractos[pt][condit]['R'])
#%%
img = [None] * len(DO_positive)
for aa,amalg in enumerate(DO_positive):
    img[aa] = tractos[DO_positive[aa][0]][DO_positive[aa][1]][DO_positive[aa][2]]
    
condit_avg[condit] = image.math_img("np.mean(np.array([img1,img2,img3,img4,img5,img6]),axis=0)",img1=img[0],img2=img[1],img3=img[2],img4=img[3],img5=img[4],img6=img[5])
plotting.plot_glass_brain(condit_avg[condit],black_bg=True,title='DO Positives',vmin=0,vmax=2)


#%% Now DO Negative
img = [None] * len(DO_negative)
math_string = ''
for aa,amalg in enumerate(DO_negative):
    img[aa] = tractos[DO_negative[aa][0]][DO_negative[aa][1]][DO_negative[aa][2]]
    math_string += 'img' + str(aa) + ','

iter_imgs = {'img'+str(num):img[num] for num in range(len(DO_negative))}

condit_avg[condit] = image.math_img("np.mean(np.array(["+math_string+"]),axis=0)",**iter_imgs)
plotting.plot_glass_brain(condit_avg[condit],black_bg=True,title='DO Negatives',vmin=0,vmax=2)


#%%
#Find the mean for a condit
condit_avg = nestdict()
for cc, condit in enumerate(['OnT','OffT']):
    #condit_avg[condit] = image.math_img("np.mean(np.array([img4,img5,img6]),axis=0)",img1=combined['901'][condit],img2=combined['903'][condit],img3=combined['905'][condit],img4=combined['906'][condit],img5=combined['907'][condit],img6=combined['908'][condit])
    condit_avg[condit] = image.math_img("np.median(np.array([img5,img6]),axis=0)",img1=combined['901'][condit],img2=combined['903'][condit],img3=combined['905'][condit],img4=combined['906'][condit],img5=combined['907'][condit],img6=combined['908'][condit])
    #condit_avg[condit] = nilearn.image.math_img("img1+img2+img3+img4+img5+img6",img1=combined['901'][condit],img2=combined['903'][condit],img3=combined['905'][condit],img4=combined['906'][condit],img5=combined['907'][condit],img6=combined['908'][condit])
    plotting.plot_glass_brain(condit_avg[condit],black_bg=True,title=condit + ' average tractography',vmin=0,vmax=2)
    #plotting.plot_glass_brain(nibabel.Nifti1Image(np.median(np.sum(data_arr[:,0,:,:,:,:],axis=2),axis=0),affine=np.eye(4)))
#%%
diff_map = nestdict()
diff_map['OnT'] = image.math_img("(img1-img2) > 0.1",img1=condit_avg['OnT'],img2=condit_avg['OffT'])
diff_map['OffT'] = image.math_img("(img1-img2) < -0.1",img1=condit_avg['OnT'],img2=condit_avg['OffT'])

for target in ['OnT','OffT']:
    #plt.figure()
    #voxels = np.array(condit_avg[target].dataobj)
    #plt.hist(voxels.flatten(),bins=200,range=(0,1))
    
    plotting.plot_glass_brain(diff_map[target],black_bg=True,title=target + ' Preference Flow',vmin=-1,vmax=1)#,threshold=0.3)
    
    #test = np.mean(np.array(dti_file),axis=0)
    #plotting.plot_img(test)
    plt.show()
