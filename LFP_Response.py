#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 22:05:44 2018

@author: virati
LFP Response Script
"""

import sys
sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/')
import DBSpace as dbo
from DBSpace import nestdict

from stream_dEEG import streamLFP

import matplotlib.pyplot as plt

# Which two epochs are we analysing?
win_list = ['Bilat','PreBilat']

#%%
eg_rec = streamLFP(pt='906',condit='OnT')
rec = eg_rec.time_series(epoch_name='PreBilat')

# The end structure we want
