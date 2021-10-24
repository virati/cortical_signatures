#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 22:05:44 2018

@author: virati
Network Action - Compare ONT vs OFFT for SCC-LFP
"""

import sys
sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/')
import DBSpace as dbo
from DBSpace import nestdict
from DBSpace.control import network_action

import itertools
from itertools import product as cart_prod

import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

import copy
from copy import deepcopy

do_pts = ['901','903','905','906','907','908']
analysis = network_action.local_response(do_pts = do_pts)
analysis.extract_baselines()
analysis.extract_response()
analysis.gen_osc_distr()

#%%
#Results plotting

analysis.plot_responses(do_pts=do_pts)

analysis.plot_patient_responses()


analysis.plot_segment_responses(do_pts = do_pts)

