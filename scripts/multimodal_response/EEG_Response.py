# %%
%load_ext autoreload
%autoreload 2
#%%
# Confirmed Working 3/29/2025

import dbspace as dbo
from dbspace.control import proc_dEEG

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("paper")
sns.set(font_scale=3)
sns.set_style("white")
# %%
pt_list = ["906", "907", "908"]
do_condits = ["OnT", "OffT"]

## Basic initialization methods, need to suppress figures from these and clean these up
eFrame = proc_dEEG.proc_dEEG(pts=pt_list, procsteps="conservative", condits=do_condits)
eFrame.standard_pipeline(blank_out_gamma=False)

# %% PSD plotting
#eFrame.plot_psd(pt="907", condit="OnT", epoch="BONT")  #'Off_3')

# %%
# Channel-marginalized Response Histogram
for pt in pt_list:
    eFrame.pop_meds(response=True, pt=pt)
    eFrame.band_distr(do_moment="mads")
    plt.suptitle(pt)

eFrame.pop_meds(response=True,pt='POOL', seg_lim=(0,10))
eFrame.band_distr(do_moment="mads")

# %%%
eFrame.topo_median_response(
    do_condits=do_condits, pt="POOL", band="Alpha", use_maya=False, seg_lim=(0,10)
)
