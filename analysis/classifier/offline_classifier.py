# %%
%load_ext autoreload
%autoreload 2
"""
Created on Sun Jan 13 20:25:09 2019

@author: virati
clClassif Script
Binary Classification for Cleaned EEG Data
"""

# %%
from dbspace.control.offline_segments import network_action_dEEG

# %%
all_pts = ["906", "907", "908"]

EEG_analysis = network_action_dEEG(
    pts=all_pts,
    procsteps="conservative",
    condits=["OnT", "OffT"],
    config_file=None,
)
# %%
# Run the basic pipeline
EEG_analysis.standard_pipeline()
EEG_analysis.train_binSVM(mask=False)
# EEG_analysis.new_SVM_dsgn(do_plot=True)
EEG_analysis.oneshot_binSVM()
EEG_analysis.bootstrap_binSVM()
EEG_analysis.analyse_binSVM(feature_weigh=False)

# EEG_analysis.OnT_dr(data_source=EEG_analysis.SVM_coeffs)
# %%
EEG_analysis.learning_binSVM()
