# %%
%load_ext autoreload
%autoreload 2
#%%

from dbspace.control import proc_dEEG

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("paper")
sns.set(font_scale=1)
sns.set_style("white")
# %%
pt_list = ["906", "907", "908"]
do_condits = ["OnT", "OffT"]

## Basic initialization methods, need to suppress figures from these and clean these up
eFrame = proc_dEEG.proc_dEEG(pts=pt_list, procsteps="conservative", condits=do_condits)
eFrame.standard_pipeline(blank_out_gamma=False)

#%%
for band in ['Alpha','Beta*']:
    eFrame.topo_median_response(do_condits=['OnT'],band=band,render_3d=True)

#%%
eFrame.topo_OnT_actionmode(pt='POOL',do_plot=True,render_3d=True)

#%%
# This one focuses on a single oscillatory band and tracks channels that 'change together'
eFrame.topo_OnT_alpha_ctrl(pt='POOL',do_plot=True,band='Alpha')