#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 01:27:54 2018

@author: virati
Simplified Harmonics Calculator
"""

import numpy as np
import matplotlib.pyplot as plt

stim_freq = 130
samp_freq = 422

fspace = np.linspace(-1000,1000,500)
stim_idxs = np.array(np.where(np.logical_and(fspace < 131,fspace > 129)) + np.where(np.logical_and(fspace > -131,fspace < -129)))

samp_comb = np.zeros_like(fspace)
#samp_idxs = np.array(np.where(np.logical_and(fspace < samp_freq+1,fspace > samp_freq-1)) + np.where(np.logical_and(fspace > -131,fspace < -129)))


freqpow = np.zeros_like(fspace)
freqpow[stim_idxs] = 1

plt.figure()
plt.stem(fspace,freqpow)
plt.show()