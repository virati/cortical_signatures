#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 21:24:09 2018

@author: virati
dEEG Continuous
Load in continuous, raw dEEG from the mat converted files
"""

import scipy.io as scio
import numpy as np
import pandas as pds
from collections import defaultdict

import pdb

Targeting = defaultdict(dict)
Targeting['All'] = {'908':
    {'OnT':
        {'fname':'/home/virati/MDD_Data/proc_hdEEG/hdEEG_MATs/DBS908/B4/DBS908_TurnOn_Day1_onTARGET.mat'},
    'OffT':
        {'fname':'/home/virati/MDD_Data/proc_hdEEG/hdEEG_MATs/DBS908/B4/DBS908_TurnOn_Day2_offTARGET.mat'}
        }}


class streamEEG:

    do_pts = ['908']
    def __init__(self):
        self.data_dict = {ev:0 for ev in self.do_pts}
        
        for pt in self.do_pts:
            for condit in ['OnT','OffT']:
                container = scio.loadmat(Targeting['All'][pt][condit]['fname'])
                dkey = [key for key in container.keys() if key[0:3] == 'DBS']
                print(dkey)
                pdb.set_trace()
                self.data_dict[pt] = np.zeros((6*60*1000,257))
                for cc in range(len(container[dkey])):
                    
                    self.data_dict[pt][:,cc] = container[dkey][9 * 60 * 1000:15*60*1000]
                    
                del(container)
                
                
sEEG = streamEEG()

        

