#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 22:36:06 2018

@author: virati
Bring in and view tractography
Focus on differentiating ONTarget and OFFTarget

Generates Figures   
"""

#%%

import dbspace as dbo
import dbspace.control.DTI as DTI
import matplotlib.pyplot as plt
import numpy as np
from dbspace.utils.structures import nestdict
from nilearn import image, plotting

Etrode_map = DTI.Etrode_map

# do_pts = ["906", "907", "908"]  #
do_pts = ["901", "903", "905", "906", "907", "908"]

vrange = range(2, 7)

dti_file = nestdict()
data = nestdict()
data_arr = np.zeros((6, 2, 2, 182, 218, 182))
combined_LR = nestdict()

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

        combined_LR[pt][condit] = image.math_img(
            "img1+img2", img1=all_vs[pp][condit]["L"], img2=all_vs[pp][condit]["R"]
        )


#%%
# Find the mean for a condit


def tracto_vars(ptlist, condit="OnT"):

    pt_pos_string = ""
    pt_pos_imgs = [None] * len(ptlist)

    for pp, pt in enumerate(ptlist):
        pt_pos_imgs[pp] = combined_LR[pt][condit]
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
thresh = 0.05
diff_map["OnT"] = image.math_img(
    "img1 > img2+" + str(thresh), img1=pt_pos["OnT"], img2=pt_pos["OffT"]
)
diff_map["OffT"] = image.math_img(
    "img2 > img1+" + str(thresh), img1=pt_pos["OnT"], img2=pt_pos["OffT"]
)

#%% Plot the difference map between targets
for target in ["OnT", "OffT"]:
    plotting.plot_glass_brain(
        diff_map[target],
        black_bg=True,
        title=target + " Engaged Preference Mask",
        vmin=-1,
        vmax=1,
    )

    plt.show()
#%% Plot the masked version
for target in ["OnT", "OffT"]:

    final = image.math_img("img1 * img2", img1=diff_map[target], img2=pt_pos[target])
    plotting.plot_glass_brain(
        final, black_bg=True, title=target + " Engaged Tractography", vmin=-1, vmax=1
    )
