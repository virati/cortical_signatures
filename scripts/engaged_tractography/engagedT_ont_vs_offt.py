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
all_DTI.plot_engaged_tractography(condits = ["OnT"])
#%%

for condition in [["OnT"], ["OffT"]]:
    all_DTI.plot_engaged_tractography(condits=condition, export_files = True)#%%

#%%
preference_threshold = 0.9
all_DTI.plot_preference_mask(threshold=preference_threshold)

#%%
all_DTI.calculate_preference_mask(
    condits=["OnT", "OffT"],
    threshold=preference_threshold,
    export_file=True,
)

#%%
all_DTI.plot_preference_diff(
    condits=["OnT", "OffT"]
)