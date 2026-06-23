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

configurations = {
    "Bilateral": ("BONT", "BOFFT"),
    "Left":      ("LONT", "LOFFT"),
    "Right":     ("RONT", "ROFFT"),
}
do_condits = [c for pair in configurations.values() for c in pair]
stim_on_condits = [ont for ont, _ in configurations.values()]

eFrame = proc_dEEG.proc_dEEG(pts=pt_list, procsteps="conservative", condits=do_condits)
eFrame.standard_pipeline(blank_out_gamma=False)

#%%
for band in ['Alpha', 'Beta*']:
    for condit in stim_on_condits:
        eFrame.topo_median_response(do_condits=[condit], band=band, render_3d=True, write_output='/tmp/cort_response/')

#%%
for condit in stim_on_condits:
    eFrame.topo_OnT_actionmode(pt='POOL', do_plot=True, render_3d=True, do_condits=[condit])

#%%
# Tracks channels that 'change together' per stimulation configuration
for condit in stim_on_condits:
    eFrame.topo_OnT_alpha_ctrl(pt='POOL', do_plot=True, band='Alpha', do_condits=[condit])
