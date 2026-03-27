#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 22:36:06 2018

@author: virati
Bring in and view tractography
Focus on differentiating ONTarget and OFFTarget

This one now looks at the DO tractography at granularity of left/right, Ont/OffT, patients, etc.
"""

#%%

import itertools

import dbspace.control.DTI as DTI
import matplotlib.pyplot as plt
import numpy as np
from dbspace.utils.structures import nestdict
from nilearn import image, plotting

all_pts = ["901", "903", "905", "906", "907", "908"]
all_condits = ["OnT", "OffT"]
all_sides = ["L", "R", "L+R"]
voltage = "2"

DO_all = itertools.product(all_pts, all_condits, all_sides)

dti_file = nestdict()
data = nestdict()
tractos = nestdict()

data_arr = np.zeros((6, 2, 2, 182, 218, 182))
combined = nestdict()

for pp, pt in enumerate(all_pts):
    for cc, condit in enumerate(["OnT", "OffT"]):
        for ss, side in enumerate(["L", "R"]):
            cntct = Etrode_map[condit][pt][ss] + 1
            dti_file[pp][condit][side] = (
                "/home/virati/Dropbox/projects/Research/MDD-DBS/Data/Anatomy/DTI/MDT_DBS_2_7V_Tractography/DBS"
                + str(pt)
                + "."
                + side
                + str(cntct)
                + "."
                + voltage
                + "V.bin.nii.gz"
            )

            tractos[pt][condit][side] = image.smooth_img(
                dti_file[pp][condit][side], fwhm=1
            )

            data_arr[pp, cc, ss, :, :, :] = np.array(tractos[pt][condit][side].dataobj)

        tractos[pt][condit]["L+R"] = image.math_img(
            "img1+img2", img1=tractos[pt][condit]["L"], img2=tractos[pt][condit]["R"]
        )

#%%
do_feel, feel_condit = feel_negative, "Negative"
do_feel, feel_condit = feel_positive, "Positive"

for do_feel, feel_condit in [
    (feel_negative, "Negative"),
    (feel_positive, "Positive"),
    (feel_weird, "Weird"),
]:

    img = [None] * len(do_feel)
    feel_string = ""
    for aa, amalg in enumerate(do_feel):
        img[aa] = tractos[do_feel[aa][0]][do_feel[aa][1]][do_feel[aa][2]]
        feel_string += "img" + str(aa) + ","

    feel_pos = nestdict()

    iter_feel_pos = {"img" + str(num): img[num] for num in range(len(do_feel))}

    feel_pos[condit] = image.math_img(
        "np.mean(np.array([" + feel_string + "]),axis=0)", **iter_feel_pos
    )
    plotting.plot_glass_brain(
        feel_pos[condit], black_bg=True, title="Feel " + feel_condit, vmin=0, vmax=2
    )


#%% Subtract the two somehow
diff_map = image.math_img(
    "img0 - img1 < -0.1", img0=do_pos[condit], img1=do_neg[condit]
)
plotting.plot_glass_brain(diff_map, black_bg=True, title="DO Neg More", vmin=-2, vmax=2)


diff_map = image.math_img("img0 - img1 > 0.1", img0=do_pos[condit], img1=do_neg[condit])
plotting.plot_glass_brain(diff_map, black_bg=True, title="DO Pos More", vmin=-2, vmax=2)
