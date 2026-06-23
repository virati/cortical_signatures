# %%
%load_ext autoreload
%autoreload 2
#%%
import dbspace.control.DTI as DTI

base_data_dir = "/home/virati/Data/phd_vrt_2013/neural/imaging/DTI/"

do_pts = ["906", "907", "908"]
electrode_map = "../../assets/experiments/metadata/mayberg_900S_electrode_map.json"

#%%
all_DTI = DTI.engaged_tractography(
    do_pts=do_pts,
    target_electrode_map=electrode_map,
    base_data_dir=base_data_dir,
)
all_DTI.load_dti(hide_progress=False)

#%%
all_DTI.plot_dti_voltage(pt="908", condit="OnT")

#%%
# Engaged tractography for each stimulation condition vs OffT
for condit in ["OnT", "LeftT", "RightT", "OffT"]:
    all_DTI.plot_engaged_tractography(condits=[condit])
    all_DTI.plot_engaged_tractography(condits=[condit], mean_op="median")

#%%
for condition in [["OnT"], ["LeftT"], ["RightT"], ["OffT"]]:
    all_DTI.plot_engaged_tractography(condits=condition, export_files=True)

#%%
preference_threshold = 0.9

# Bilateral vs off
all_DTI.plot_preference_mask(threshold=preference_threshold)

# Lateralized preference comparisons
for active_condits in [["OnT", "OffT"], ["LeftT", "OffT"], ["RightT", "OffT"]]:
    all_DTI.calculate_preference_mask(
        condits=active_condits,
        threshold=preference_threshold,
        export_file=True,
    )
    all_DTI.plot_preference_diff(condits=active_condits)
    all_DTI.plot_preference_level(condits=active_condits)

#%%
# Left vs Right direct comparison
all_DTI.plot_preference_diff(condits=["LeftT", "RightT"])
all_DTI.plot_preference_level(condits=["LeftT", "RightT"])
