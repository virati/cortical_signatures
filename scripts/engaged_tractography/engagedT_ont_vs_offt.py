# %%
import dbspace.control.DTI as DTI

base_data_dir = "/home/virati/Data/phd_vrt_2013/neural/imaging/DTI/"

do_pts = ["901", "903", "905", "906", "907", "908"]
all_DTI = DTI.engaged_tractography(
    do_pts=do_pts,
    target_electrode_map="../../assets/experiments/metadata/Electrode_Map.json",
)
all_DTI.load_dti(hide_progress=False)
