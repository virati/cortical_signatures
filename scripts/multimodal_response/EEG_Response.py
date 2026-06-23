# %%
%load_ext autoreload
%autoreload 2
#%%
# Confirmed Working 3/29/2025

import dbspace as dbo
from dbspace.control import proc_dEEG

import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
import tempfile

import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


sns.set_context("paper")
sns.set(font_scale=2)
sns.set_style("white")
# %%
pt_list = ["906", "907", "908"]

configurations = {
    "Bilateral": ("BONT", "BOFFT"),
    "Left":      ("LONT", "LOFFT"),
    "Right":     ("RONT", "ROFFT"),
}
do_condits = [c for pair in configurations.values() for c in pair]

eFrame = proc_dEEG.proc_dEEG(pts=pt_list, procsteps="liberal", condits=do_condits)
eFrame.standard_pipeline(blank_out_gamma=False)

# %% PSD plotting
#eFrame.plot_psd(pt="907", condit="BONT", epoch="BONT")

# %%
# Channel-marginalized Response Histogram — all conditions
for pt in pt_list:
    eFrame.pop_meds(response=True, pt=pt)
    eFrame.plot_band_distr(do_moment="mads")
    plt.suptitle(pt)

eFrame.pop_meds(response=True, pt='POOL', seg_lim=(0, 10))
eFrame.band_distr(do_moment="mads")
