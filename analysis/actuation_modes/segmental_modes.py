#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 17:06:33 2019

@author: virati
Separate script just for control mode analysis of ONT (and OFFT)
"""


from dbspace.control import stream_buffers
import seaborn as sns

sns.set_context("paper")
sns.set(font_scale=3)
sns.set_style("white")
#%%

## Basic initialization methods, need to suppress figures from these and clean these up
eFrame = stream_buffers.proc_dEEG(
    pts=["906", "907", "908"], procsteps="conservative", condits=["OnT", "OffT"]
)
eFrame.standard_pipeline()

# eFrame.topo_OnT_ctrl_tensor(pt='POOL')

#%%
# This one focuses on a single oscillatory band and tracks channels that 'change together'
eFrame.topo_OnT_alpha_ctrl(pt="POOL", do_plot=True, band="Alpha")
# Plot the \alpha specific control across time
eFrame.plot_alpha_ctrl_L(top_comp=5)
eFrame.plot_alpha_ctrl_S(top_comp=1)
