#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 21:32:14 2018

@author: virati
make spectrograms for the streaming EEG
"""

from stream_dEEG import streamEEG


import sys
sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/')
import DBS_Osc as dbo
from DBS_Osc import nestdict

import itertools
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import numpy as np

pt = '901'
condit = 'OnT'

sEEG = streamEEG(ds_fact=2,pt=pt,condit=condit,spotcheck=True)