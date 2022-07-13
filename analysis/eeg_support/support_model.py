#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 19:55:19 2019

@author: virati
Main script for forward modeling
"""

#%%

from dbspace.control import segmented_dEEG
from dbspace.control.DTI_support import (
    DTI_support_model,
    plot_support_model,
    plot_EEG_masks,
)

#%%
pt_list = ["906", "907", "908"]
condit = "OnT"
#%%
## Basic initialization methods, need to suppress figures from these and clean these up
eFrame = segmented_dEEG.network_action_dEEG(
    pts=pt_list, procsteps="conservative", condits=[condit]
)
eFrame.standard_pipeline()
eFrame.OnT_ctrl_dyn(condit=condit)

#%%
# Virtual Dissection Here

for band in ["Alpha"]:
    for pt in pt_list:
        EEG_support = DTI_support_model(
            pt,
            4,
            dti_parcel_thresh=20,
            eeg_thresh=50,
            electrode_map_file="../../assets/experiments/metadata/Electrode_Map.json",
        )
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
