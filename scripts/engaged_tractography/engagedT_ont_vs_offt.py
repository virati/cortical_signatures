# %%
%load_ext autoreload
%autoreload 2
#%%
import dbspace.control.DTI as DTI

base_data_dir = "/home/virati/Data/phd_vrt_2013/neural/imaging/DTI/"

do_pts = ["906", "907", "908"]
electrode_map = "../../assets/experiments/metadata/Electrode_Map.json"

#%%
all_DTI = DTI.engaged_tractography(
    do_pts=do_pts,
    target_electrode_map=electrode_map,
    base_data_dir=base_data_dir,
)
all_DTI.load_dti(hide_progress=False)

for condition in ["OnT", "OffT"]:

    all_DTI.plot_engaged_tractography(condits=condition)#%%

#%%

all_DTI.plot_preference_mask(threshold=0.5)