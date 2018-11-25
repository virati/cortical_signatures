#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 20:06:36 2018

@author: virati
Tractography Class to preprocess and package DTI data relevant to project
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

import DBSpace as dbo
from DBSpace import nestdict



class DTI:
    def __init__(self,do_pts=['901','903','905','906','907','908']):
        self.do_pts = do_pts
        self.v_list = [2,3,4,5,6,7]

    def load_data(self):
        dti_file = nestdict()
        data = nestdict()
        data_arr = np.zeros((1,1,6,2,182,218,182))
        combined = nestdict()

        for pp,pt in enumerate(self.do_pts):
            for cc,condit in enumerate(['OnT']):
                for vv,vstim in enumerate(self.v_list):
                    for ss,side in enumerate(['L','R']):
                        cntct = dbo.Etrode_map[condit][pt][ss]+1
                        fname = '/home/virati/Dropbox/projects/Research/MDD-DBS/Data/Anatomy/DTI/MDT_DBS_2_7V_Tractography/DBS'+str(pt) + '.'+side+str(cntct)+'.'+str(vstim) + 'V.bin.nii.gz'
                        dti_file[pt][condit][vstim][side] = fname
                    
                        #data[pt][condit][vstim][side] = image.smooth_img(dti_file[pt][condit][vstim][side],fwhm=1)
                        data[pt][condit][vstim][side] = image.load_img(dti_file[pt][condit][vstim][side])
                        
                        data_arr[pp,cc,vv,ss,:,:,:] = np.array(data[pt][condit][vstim][side].dataobj)
                    combined[pt][condit][vstim] = image.math_img("img1+img2",img1=data[pt][condit][vstim]['L'],img2=data[pt][condit][vstim]['R'])
                    
                stim_mask = np.sum(data_arr,axis=3).squeeze()
        
                middle = (stim_mask>0).astype(np.int)
                middle_idx = np.argmax(middle,axis=0)
                
        self.combined = combined
        self.stim_mask = stim_mask
        self.middle_idx = middle_idx
        self.data = data

    def plot_V_thresh(self,pt='906',condit='OnT'):
        #%%
        # plot the 
        new_img = nilearn.image.new_img_like(self.data[pt][condit][vstim]['L'],(self.middle_idx))
        plotting.plot_glass_brain(new_img)
        
    '''
    This method plots the DTI for a given patient x condition combination
    '''
    def plot_V_DTI(self,pt='906',condit='OnT',v_select = 2,merged=False):
        combined = self.combined
        vidx = self.v_list.index(v_select)
        
        condit = 'OnT'
        
        if merged:
            stacked = image.math_img("img1+img2+img3+img4+img5+img6",img1=combined[pt][condit][2],img2=combined[pt][condit][3],img3=combined[pt][condit][4],img4=combined[pt][condit][5],img5=combined[pt][condit][6],img6=combined[pt][condit][7])
        
            plotting.plot_glass_brain(stacked,black_bg=True,title=condit + ' Tractography',vmin=-15,vmax=15)
        else:
            new_img = nilearn.image.new_img_like(self.data[pt][condit][v_select]['L'],(self.middle_idx))
            plotting.plot_glass_brain(new_img)



V_DTI = DTI(do_pts=['906'])
V_DTI.load_data()
V_DTI.plot_V_DTI()