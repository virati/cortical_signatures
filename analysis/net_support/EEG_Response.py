#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 16:32:20 2018

@author: virati
Script to generate the EEG Response Violinplots

"""
#%%

from dbspace.control import segmented_dEEG
from dbspace.viz.MM import EEG_Viz

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("paper")
sns.set(font_scale=3)
sns.set_style("white")

pt_list = ["906", "907", "908"]

# The feature vector, in this case the frequencies
fvect = np.linspace(0, 500, 513)
do_coherence = False

## Basic initialization methods, need to suppress figures from these and clean these up
eFrame = segmented_dEEG.network_action_dEEG(
    pts=pt_list, procsteps="conservative", condits=["OnT", "OffT"]
)
eFrame.standard_pipeline()
eFrame.band_distrs()
