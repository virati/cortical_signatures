# %%
%load_ext autoreload
%autoreload 2
#%%
import dbspace.control.DTI as DTI

base_data_dir = "/home/virati/Data/phd_vrt_2013/neural/imaging/DTI/"

do_pts = ["901", "903", "905", "906", "907", "908"]
all_DTI = DTI.engaged_tractography(
    do_pts=do_pts,
    target_electrode_map="../../assets/experiments/metadata/Electrode_Map.json",
    base_data_dir=base_data_dir,
)
all_DTI.load_dti(hide_progress=False)

#%%
all_DTI.plot_engaged_tractography(condits=["OnT"])
#%%
all_DTI.plot_engaged_tractography(condits=["OffT"])
#%%
all_DTI.plot_preference_mask(threshold=0.5)

#%%
eeg_pts = ["906","907","908"] #These are the patients with both LFP and dEEG during Targeting Experiment
eeg_DTI = DTI.engaged_tractography(do_pts=eeg_pts,target_electrode_map='../../assets/experiments/metadata/Electrode_Map.json', base_data_dir=base_data_dir,)
eeg_DTI.load_dti()
eeg_DTI.plot_preference_mask(threshold=0.5)