#%%
%load_ext autoreload
%autoreload 2
#%%
from dbspace.control import network_action
from dbspace.control import proc_dEEG

import matplotlib.pyplot as plt

import logging
log = logging.getLogger(__name__)
log.info("Loading modules...")

pt_list = ["901", "903", "905", "906", "907", "908"]
do_condits = ["OnT", "OffT"]

# %%
log.info('Loading EEG Data...')
eeg_response = proc_dEEG.proc_dEEG(
    pts=['906','907','908'], procsteps="conservative", condits=do_condits
)
eeg_response.standard_pipeline(blank_out_gamma=False)

#%%
log.info('Loading LFP Data...')
lfp_response = network_action.local_response(do_pts=pt_list)
lfp_response.extract_baselines()
lfp_response.extract_response()
lfp_response.gen_osc_distr()

# %%
lfp_response.plot_responses(do_pts=pt_list)
eeg_response.plot_meds()