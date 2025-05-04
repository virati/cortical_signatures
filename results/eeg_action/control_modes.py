# %%
%load_ext autoreload
%autoreload 2
# Confirmed working 3/29/2025

from dbspace.control import offline_segments


## Basic initialization methods, need to suppress figures from these and clean these up
eFrame = offline_segments.network_action_dEEG(
    pts=["906", "907", "908"], procsteps="conservative", condits=["OnT", "OffT"]
)
eFrame.standard_pipeline()
# %%
## Let's plot all the bands first
for band in ["Alpha", "Beta*"]:
    eFrame.topo_median_response(do_condits=["OnT"], band=band, use_maya=True)

# %%
eFrame.topo_OnT_ctrl(pt="POOL", do_plot=True, plot_maya=True)
