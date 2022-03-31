#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 22:05:44 2018

@author: virati
Network Action - Compare ONT vs OFFT for SCC-LFP

"""

#%%
from dbspace.utils.structures import nestdict
from dbspace.control import network_action

do_pts = ["901", "903", "905", "906", "907", "908"]
analysis = network_action.local_response(
    config_file="../../assets/config/stream_config.json", do_pts=do_pts
)
analysis.extract_baselines()
analysis.extract_response()
analysis.gen_osc_distr()

#%%
# Results plotting

analysis.plot_responses(do_pts=do_pts)

analysis.plot_patient_responses()


analysis.plot_segment_responses(do_pts=do_pts)
