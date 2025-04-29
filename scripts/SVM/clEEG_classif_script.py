# %%
%load_ext autoreload
%autoreload 2
# Confirmed Working 3/29/2025
from dbspace.control import proc_dEEG

# from proc_dEEG import proc_dEEG
import seaborn as sns

sns.set_context("paper")
sns.set(font_scale=4)
sns.set_style("white")

all_pts = ["906", "907", "908"]

EEG_analysis = proc_dEEG.proc_dEEG(
    pts=all_pts, procsteps="conservative", condits=["OnT", "OffT"]
)
# %%
EEG_analysis.standard_pipeline()
EEG_analysis.train_binary_svm(mask=False)

#%%
EEG_analysis.oneshot_binSVM()
EEG_analysis.bootstrap_binSVM()
EEG_analysis.analyse_binSVM(plotting=True, analysis_approach='raw')

# EEG_analysis.OnT_dr(data_source=EEG_analysis.SVM_coeffs)
# %%
# Learning Curve for the Binary SVM
EEG_analysis.learning_binSVM()
