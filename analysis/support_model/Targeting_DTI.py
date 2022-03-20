#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 22:36:06 2018

@author: virati
Bring in and view tractography
Focus on differentiating ONTarget and OFFTarget

OBSOLETE
"""

#%%

import numpy as np
import nibabel
import nilearn
from nilearn import plotting, image
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn import surface
import os

from mpl_toolkits.mplot3d import Axes3D
import dbspace as dbo
from dbspace.utils.structures import nestdict

import dbspace.control.DTI as DTI

Etrode_map = DTI.Etrode_map

do_pts = ["906", "907", "908"]  # ['901','903','905','906','907','908']
voltage = "2"
vrange = range(2, 7)

dti_file = nestdict()
data = nestdict()
data_arr = np.zeros((6, 2, 2, 182, 218, 182))
combined = nestdict()

# fsaverage = datasets.fetch_surf_fsaverage5()

# load_file = nestdict()
all_vs = nestdict()

for pp, pt in enumerate(do_pts):
    for cc, condit in enumerate(["OnT", "OffT"]):
        for ss, side in enumerate(["L", "R"]):
            vv_pos_imgs = [None] * len(vrange)
            vv_pos_string = ""
            for vv, volt in enumerate(vrange):
                cntct = Etrode_map[condit][pt][ss] + 1
                dti_file[pp][condit][side] = (
                    "/home/virati/Dropbox/projects/Research/MDD-DBS/Data/Anatomy/DTI/MDT_DBS_2_7V_Tractography/DBS"
                    + str(pt)
                    + "."
                    + side
                    + str(cntct)
                    + "."
                    + str(volt)
                    + "V.bin.nii.gz"
                )
                vv_pos_imgs[vv] = image.smooth_img(dti_file[pp][condit][side], fwhm=1)
                vv_pos_string += "img" + str(vv) + ","

            iter_do_vvs = {
                "img" + str(num): vv_pos_imgs[num] for num in range(len(vrange))
            }

            all_vs[pp][condit][side] = image.math_img(
                "np.mean(np.array([" + vv_pos_string + "]),axis=0)", **iter_do_vvs
            )

            # data[pp][condit][side] = image.smooth_img(dti_file[pp][condit][side],fwhm=1)
            data_arr[pp, cc, ss, :, :, :] = np.array(all_vs[pp][condit][side].dataobj)

        combined[pt][condit] = image.math_img(
            "img1+img2", img1=all_vs[pp][condit]["L"], img2=all_vs[pp][condit]["R"]
        )


#%%
# Find the mean for a condit


def tracto_vars(ptlist, condit="OnT"):

    pt_pos_string = ""
    pt_pos_imgs = [None] * len(ptlist)

    for pp, pt in enumerate(ptlist):
        pt_pos_imgs[pp] = combined[pt][condit]
        pt_pos_string += "img" + str(pp) + ","

    iter_do_pts = {"img" + str(num): pt_pos_imgs[num] for num in range(len(ptlist))}

    return pt_pos_string, pt_pos_imgs, iter_do_pts


pt_pos = nestdict()
pt_pos_imgs = nestdict()
pt_var = nestdict()

for cc, condit in enumerate(["OnT", "OffT"]):
    pt_pos_string, pt_pos_imgs[condit], iter_do_pts = tracto_vars(do_pts, condit=condit)

    pt_pos[condit] = image.math_img(
        "np.mean(np.array([" + pt_pos_string + "]),axis=0)", **iter_do_pts
    )
    plotting.plot_glass_brain(
        pt_pos[condit], black_bg=True, title=condit, vmin=0, vmax=2
    )

    pt_var[condit] = image.math_img(
        "np.var(np.array([" + pt_pos_string + "]),axis=0)", **iter_do_pts
    )
    plotting.plot_glass_brain(
        pt_var[condit], black_bg=False, title=condit + " variance"
    )
#%%
diff_map = nestdict()
# diff_map['OnT'] = image.math_img("(img1-img2) > 0.1",img1=pt_pos['OnT'],img2=pt_pos['OffT'])
# diff_map['OffT'] = image.math_img("(img1-img2) < -0.1",img1=pt_pos['OnT'],img2=pt_pos['OffT'])

thresh = 0.05
diff_map["OnT"] = image.math_img(
    "img1 > img2+" + str(thresh), img1=pt_pos["OnT"], img2=pt_pos["OffT"]
)
diff_map["OffT"] = image.math_img(
    "img2 > img1+" + str(thresh), img1=pt_pos["OnT"], img2=pt_pos["OffT"]
)

import nilearn.input_data.nifti_masker as nifti_masker

for target in ["OnT", "OffT"]:
    # plt.figure()
    # voxels = np.array(condit_avg[target].dataobj)
    # plt.hist(voxels.flatten(),bins=200,range=(0,1))

    plotting.plot_glass_brain(
        diff_map[target],
        black_bg=True,
        title=target + " Preference Flow",
        vmin=-1,
        vmax=1,
    )  # ,threshold=0.3)

    # test = np.mean(np.array(dti_file),axis=0)
    # plotting.plot_img(test)
    plt.show()

    # nibabel.save(targ_mask,'mask.nii.gz')
    # targ_mask.to_filename('test.nii.gz')
    # nibabel.save(pt_pos[target], target + "_mask.nii.gz")
#%%
# from nilearn.plotting import plot_roi, show
# plot_roi(diff_map['OnT'])

#%%
# Now apply the mask to the raw tracto for pretty pictures!

for target in ["OnT", "OffT"]:

    final = image.math_img("img1 * img2", img1=diff_map[target], img2=pt_pos[target])
    plotting.plot_glass_brain(
        final, black_bg=True, title=target + " Preference Flow", vmin=-1, vmax=1
    )


#%%
# This is me trying to make a good 3d render of the same thing but...whatever

nibabel.viewers.OrthoSlicer3D(np.array(diff_map["OnT"].dataobj)).show()
from scipy.spatial import ConvexHull
from mayavi import mlab

ch = ConvexHull(np.array(diff_map[target].dataobj))
hull_ids = ch.vertices

mlab.points3d(x[hull_ids], y[hull_ids], z[hull_ids])
mlab.show()
