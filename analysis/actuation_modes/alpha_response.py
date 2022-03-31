#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 16:13:54 2018

@author: virati
This scipt is focused on characterizing the ONTarget response
Includes some DTI support modeling which should be split out

"""

#%%

from dbspace.control import offline_segments
from dbspace.viz.MM import EEG_Viz
from dbspace.control.DTI_support import DTI_support_model, plot_support_model

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("paper")
sns.set(font_scale=3)
sns.set_style("white")

pt_list = ["906", "907", "908"]
## Basic initialization methods, need to suppress figures from these and clean these up
eFrame = offline_segments.network_action_dEEG(
    pts=pt_list,
    procsteps="conservative",
    condits=["OnT", "OffT"],
    config_file="../../assets/config/targeting_experiment.json",
)
eFrame.standard_pipeline()
#%% Generate the control modes
# eFrame.OnT_ctrl_modes(pt='POOL') #THIS HAS BEEN MOVED TO control_modes.py

#%% Plot the median response
eFrame.topo_median_response(
    do_condits=["OnT", "OffT"],
    pt="POOL",
    band="Alpha",
    use_maya=True,
    scale_w_mad=False,
    avg_func=np.mean,
)
# eFrame.topo_median_variability(do_condits=['OnT','OffT'],pt='POOL',band='Alpha',use_maya=False)

#%% Plot individual patients
for pp in pt_list:
    eFrame.topo_median_response(
        do_condits=["OnT", "OffT"],
        pt=pp,
        band="Alpha",
        use_maya=True,
        scale_w_mad=False,
    )
# eFrame.topo_median_variability(do_condits=['OnT','OffT'],pt='POOL',band='Alpha',use_maya=False)
