'''
OBSOLETE
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 20:07:53 2018

@author: virati
This script does the voltages sweep analyses with the DTI images for our patients
"""

import numpy as np
import nibabel
import nilearn
import nilearn.image as image
from nilearn import plotting, image
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn import surface

from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/')
import DBSpace as dbo
from DBSpace import nestdict

all_pts = ['901','903','905','906','907','908']

Etrode_map = {'OnT':{'901':(2,1),'903':(2,2),'905':(2,1),'906':(2,2),'907':(1,1),'908':(2,1)},'OffT':{'901':(1,2),'903':(1,1),'905':(1,2),'906':(1,1),'907':(2,2),'908':(1,2)}}

dti_file = nestdict()
data = nestdict()
data_arr = np.zeros((1,1,6,2,182,218,182))
combined = nestdict()

pt_list = ['906']

v_list = [2,3,4,5,6,7]

for pp,pt in enumerate(pt_list):
    for cc,condit in enumerate(['OnT']):
        for vv,vstim in enumerate(v_list):
            for ss,side in enumerate(['L','R']):
                cntct = Etrode_map[condit][pt][ss]+1
                fname = '/home/virati/Dropbox/projects/Research/MDD-DBS/Data/Anatomy/DTI/MDT_DBS_2_7V_Tractography/DBS'+str(pt) + '.'+side+str(cntct)+'.'+str(vstim) + 'V.bin.nii.gz'
                dti_file[pt][condit][vstim][side] = fname
            
                #data[pt][condit][vstim][side] = image.smooth_img(dti_file[pt][condit][vstim][side],fwhm=1)
                data[pt][condit][vstim][side] = image.load_img(dti_file[pt][condit][vstim][side])
                
                data_arr[pp,cc,vv,ss,:,:,:] = np.array(data[pt][condit][vstim][side].dataobj)
            combined[pt][condit][vstim] = image.math_img("img1+img2",img1=data[pt][condit][vstim]['L'],img2=data[pt][condit][vstim]['R'])
            
        stim_mask = np.sum(data_arr,axis=3).squeeze()

        middle = (stim_mask>0).astype(np.int)
        middle_idx = np.argmax(middle,axis=0)
        


#%%
# Plot just an image transformed from matrix
vv = 0
new_img = nilearn.image.new_img_like(data[pt][condit][vstim]['L'],np.sum(data_arr[pp,cc,vv,:,:,:,:],axis=0))
plotting.plot_glass_brain(new_img)

#%%
# plot the 
new_img = nilearn.image.new_img_like(data[pt][condit][vstim]['L'],(middle_idx))
plotting.plot_glass_brain(new_img)


#%%
# This plots the glass brain for the actual tractos, specifically the sum
pt = pt_list[0]
condit = 'OnT'
#stacked = image.math_img("-2*img1+-3*img2+-4*img3+5*img4+6*img5+7*img6",img1=combined[pt][condit][2],img2=combined[pt][condit][3],img3=combined[pt][condit][4],img4=combined[pt][condit][5],img5=combined[pt][condit][6],img6=combined[pt][condit][7])
stacked = image.math_img("img1+img2+img3+img4+img5+img6",img1=combined[pt][condit][2],img2=combined[pt][condit][3],img3=combined[pt][condit][4],img4=combined[pt][condit][5],img5=combined[pt][condit][6],img6=combined[pt][condit][7])

plotting.plot_glass_brain(stacked,black_bg=True,title=condit + ' Tractography',vmin=-15,vmax=15)
plt.show()