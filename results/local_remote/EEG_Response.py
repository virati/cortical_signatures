# %%
%load_ext autoreload
%autoreload 2
# Confirmed Running Fully 3/29/2025
# %%

from dbspace.control.offline_segments import network_action_dEEG
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
eFrame = network_action_dEEG(
    pts=pt_list, procsteps="conservative", condits=["OnT", "OffT"]
)
eFrame.standard_pipeline()
eFrame.band_distrs()
