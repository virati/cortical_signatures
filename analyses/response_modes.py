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
do_condits = ["OnT", "OffT", "LeftT", "RightT"]

## Basic initialization methods, need to suppress figures from these and clean these up
eFrame = proc_dEEG.proc_dEEG(pts=pt_list, procsteps="conservative", condits=do_condits)
eFrame.standard_pipeline(blank_out_gamma=False)

#%%
for band in ['Alpha','Beta*']:
    for condit in ['OnT', 'LeftT', 'RightT']:
        eFrame.topo_median_response(do_condits=[condit], band=band, render_3d=True, write_output='/tmp/cort_response/')

#%%
for condit in ['OnT', 'LeftT', 'RightT']:
    eFrame.topo_OnT_actionmode(pt='POOL', do_plot=True, render_3d=True, do_condits=[condit])

#%%
# Tracks channels that 'change together' per stimulation laterality
for condit in ['OnT', 'LeftT', 'RightT']:
    eFrame.topo_OnT_alpha_ctrl(pt='POOL', do_plot=True, band='Alpha', do_condits=[condit])
