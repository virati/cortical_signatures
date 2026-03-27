# %%
%load_ext autoreload
%autoreload 2
# Confirmed Working 3/29/2025
from dbspace.control import network_action

# %%

do_pts = ["901", "903", "905", "906", "907", "908"]

#%%
analysis = network_action.local_response(do_pts=do_pts)
analysis.extract_baselines()
analysis.extract_response()
analysis.gen_osc_distr()
# %%
analysis.plot_responses(do_pts=do_pts)


# %%
# analysis.plot_patient_responses()
# %%
analysis.plot_segment_responses(do_pts=do_pts)
