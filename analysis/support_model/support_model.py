#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 19:55:19 2019

@author: virati
Main script for forward modeling
"""

#%%

from dbspace.control import segmented_dEEG
from dbspace.viz.MM import EEG_Viz
from dbspace.control.DTI_support import (
    DTI_support_model,
    plot_support_model,
    plot_EEG_masks,
)
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("paper")
sns.set(font_scale=3)
sns.set_style("white")

import mayavi
import mayavi.mlab as mlab

# assert(mayavi.__version__ == '4.7.1')

import pickle
import cmocean

plt.close("all")
mlab.close(all=True)
#%%

pt_list = ["906", "907", "908"]
condit = "OnT"
#%%
## Basic initialization methods, need to suppress figures from these and clean these up
eFrame = segmented_dEEG.network_action_dEEG(
    pts=pt_list, procsteps="conservative", condits=[condit]
)
eFrame.standard_pipeline()
#%%
eFrame.OnT_ctrl_dyn(condit=condit)
#%%
# The feature vector, in this case the frequencies
fvect = np.linspace(0, 500, 513)
do_coherence = False

#%%
# Here we do the forward modeling to do network dissection
# eFrame.pool_patients()
for band in ["Alpha"]:
    for pt in ["906"]:
        # 30, 25 is good
        EEG_support = DTI_support_model(
            pt,
            4,
            dti_parcel_thresh=20,
            eeg_thresh=50,
            electrode_map_file="../../assets/experiments/metadata/Electrode_Map.json",
        )  # 15,55 work
        plot_support_model(
            EEG_support,
            pt,
            layers=[1, 0, 0],
            electrode_map_file="../../assets/experiments/metadata/Electrode_Map.json",
        )
        plot_EEG_masks(EEG_support)
        eFrame.support_analysis(
            support_struct=EEG_support,
            condit=condit,
            pt=pt,
            band=band,
            voltage=str(3),
        )
