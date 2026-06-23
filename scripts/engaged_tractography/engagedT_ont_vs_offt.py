# %%
%load_ext autoreload
%autoreload 2
#%%
import dbspace.control.DTI as DTI

base_data_dir = "/home/virati/Data/phd_vrt_2013/neural/imaging/DTI/"

do_pts = ["906", "907", "908"]
electrode_map = "../../assets/experiments/metadata/mayberg_900S_electrode_map.json"

configurations = {
    "Bilateral": ("BONT", "BOFFT"),
    "Left":      ("LONT", "LOFFT"),
    "Right":     ("RONT", "ROFFT"),
}
do_condits = [c for pair in configurations.values() for c in pair]

#%%
all_DTI = DTI.engaged_tractography(
    do_pts=do_pts,
    target_electrode_map=electrode_map,
    base_data_dir=base_data_dir,
)
all_DTI.load_dti(hide_progress=False)

#%%
all_DTI.plot_dti_voltage(pt="908", condit="BONT")

#%%
# Engaged tractography plots for every condition
for condit in do_condits:
    all_DTI.plot_engaged_tractography(condits=[condit])
    all_DTI.plot_engaged_tractography(condits=[condit], mean_op="median")

for condit in do_condits:
    all_DTI.plot_engaged_tractography(condits=[condit], export_files=True)

#%%
preference_threshold = 0.9

# --- Each configuration vs its own OffT ---
for label, (ont, offt) in configurations.items():
    all_DTI.calculate_preference_mask(condits=[ont, offt], threshold=preference_threshold, export_file=True)
    all_DTI.plot_preference_mask(threshold=preference_threshold, condits=[ont, offt])
    all_DTI.plot_preference_diff(condits=[ont, offt])
    all_DTI.plot_preference_level(condits=[ont, offt])

#%%
# --- Each lateralized configuration vs shared bilateral OffT ---
for label, (ont, _) in configurations.items():
    if label != "Bilateral":
        all_DTI.calculate_preference_mask(condits=[ont, "BOFFT"], threshold=preference_threshold, export_file=True)
        all_DTI.plot_preference_mask(threshold=preference_threshold, condits=[ont, "BOFFT"])
        all_DTI.plot_preference_diff(condits=[ont, "BOFFT"])
        all_DTI.plot_preference_level(condits=[ont, "BOFFT"])

#%%
# --- Direct Left vs Right comparison ---
all_DTI.plot_preference_diff(condits=["LONT", "RONT"])
all_DTI.plot_preference_level(condits=["LONT", "RONT"])
