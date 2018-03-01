#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 23:17:11 2018

@author: virati
File to flatten coordinates of dEEG
"""
import mne

import matplotlib.pyplot as plt

egipos = mne.channels.read_montage('/home/virati/Dropbox/GSN-HydroCel-257.sfp')
etrodes = egipos.pos

#layout = mne.channels.read_layout('Vectorview-all')
mne.viz.plot_topomap(5*np.ones((257)),pos=etrodes[:,[0,1]],image_interp='none',outlines='skirt',contours=0,sensors=True,show_names=str(np.linspace(0,257,257)),res=1000)
#mne.viz.plot_layout(layout)
#mne.viz.plot_sensors(egipos)

#what's the distribution of Zs?
#plt.hist(etrodes[:,2],bins=100)