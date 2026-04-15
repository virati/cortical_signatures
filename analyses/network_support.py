#%%
%load_ext autoreload
%autoreload 2

# %% [markdown]
# # Network Support Analyses
# Basic support analysis for SCCwm-DBS - support here in the "mathematical" sense of where we have non-zero terms.
# Determine whether there's a meaningful response in local and/or remote recordings upon initiation of SCCwm-DBS.

#%%
from dbspace.control import network_action
from dbspace.control import proc_dEEG

import matplotlib.pyplot as plt

import logging
log = logging.getLogger(__name__)
log.info("Loading modules...")

pt_list = ["901", "903", "905", "906", "907", "908"]
do_condits = ["OnT", "OffT"]
#%% [markdown] First, let's look at the *local* response, as measured in bilateral SCC-$\partial$LFP recordings.
#%%
log.info('Loading LFP Data...')
local_response = network_action.local_response(do_pts=pt_list)
local_response.extract_baselines()
local_response.extract_response()
local_response.gen_osc_distr()

# %% [markdown] Next, let's look at the *remote* response, as measured in dense-array EEG recordings across the scalp.
#%%
log.info('Loading EEG Data...')
remote_response = proc_dEEG.proc_dEEG(
    pts=['906','907','908'], procsteps="conservative", condits=do_condits
)
remote_response.standard_pipeline(blank_out_gamma=False)


# %% [markdown] Plot the responses in both the local and remote responses.
#%%
local_response.plot_responses(do_pts=pt_list)
remote_response.plot_meds()