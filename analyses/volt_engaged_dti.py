# %%
%load_ext autoreload
%autoreload 2
#%%
import dbspace.control.DTI as DTI

base_data_dir = "/home/virati/Data/phd_vrt_2013/neural/imaging/DTI/"

do_pts = ["906", "907", "908"]
electrode_map = "../assets/experiments/metadata/mayberg_900S_electrode_map.json"

configurations = {
    "Bilateral": ("BONT", "BOFFT"),
    "Left":      ("LONT", "LOFFT"),
    "Right":     ("RONT", "ROFFT"),
}
do_condits = [c for pair in configurations.values() for c in pair]

#%%
all_DTI = DTI.engaged_tractography(
    do_pts=do_pts,
    volt_range=range(2, 8),
    do_condits=do_condits,
    target_electrode_map=electrode_map,
    base_data_dir=base_data_dir,
)
all_DTI.load_dti(hide_progress=False)

#%%
# Each configuration vs its own OffT baseline
for label, (ont, offt) in configurations.items():
    all_DTI.plot_preference_mask(threshold=1.1, condits=[ont, offt])

#%%
# Each lateralized configuration vs the shared bilateral OffT
for label, (ont, _) in configurations.items():
    if label != "Bilateral":
        all_DTI.plot_preference_mask(threshold=1.1, condits=[ont, "BOFFT"])
