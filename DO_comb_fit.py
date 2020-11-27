#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 13:37:39 2020

@author: virati
Comb filtering example

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import sys
#sys.path.append('/home/virati/Dropbox/projects/Research/DBSControl/autoDyn/')
#from dyn_sys import dyn_sys
import networkx as nx
import pdb
from allantools import noise

plt.close('all')

class delay_filt2:
    def __init__(self,s0,se=[],s_decay=0):
        pass

class delay_filt:
    def __init__(self,k=10,sec_delay=[],k_decay=0):

        self.fs = 422
        self.tvect = np.linspace(0,60,60 * self.fs)
        self.set_delay(k=k,delay=sec_delay,decay=k_decay)
        
    '''
    KEEP IN MIND delay is in seconds wrt fs
    '''
    def set_delay(self,k,delay,decay):
        if decay != 0:
            self.k = (np.round(self.fs * delay * np.ones_like(self.tvect))).astype(np.int)
        else:
            self.k = k * np.ones_like(self.tvect)
            
        self.k_decay = decay
        self.k = np.round((np.exp(-self.k_decay * self.tvect)) * self.k).astype(np.int)

    def design_oscillation(self,decay=0.1,center_freq=10,amp_decay=0,flatline=False,amplitude=1):
        tvect = self.tvect
        if flatline:
            self.inp_sig = np.zeros_like(tvect)# + np.random.normal(0,0.1,size=tvect.shape)
        else:
            center_freq = center_freq*np.exp(-decay*tvect)#;plt.plot(center_freq)
            self.inp_sig = amplitude*np.exp(-amp_decay*tvect)*(np.sin(2 * np.pi * center_freq * tvect))
            #self.inp_sig += np.random.normal(0,1e-7,size=tvect.shape)# + 1/3 * np.sin(2 * np.pi * 3*center_freq * tvect) + 0/5 * np.sin(2 * np.pi * 5*center_freq * tvect))
            #inp_sig = np.exp(-0.2*tvect)*(sig.square(2*np.pi*center_freq * tvect)) #DO PAC OF THIS
            #inp_sig = np.exp(-0.1*tvect)*sig.chirp(tvect,f0=10,t1=10,f1=2,method='hyperbolic')
            #inp_sig[0:5] = 100
    
    def run(self):
        for tt,time in enumerate(self.tvect):
            self.y[tt] = self.inp_sig[tt] + self.y[tt-self.k[tt]]# + noise.brown(1,fs=self.fs)
            #self.y[tt] = self.inp_sig[tt] + self.inp_sig[tt-self.k[tt]]# + noise.brown(1,fs=self.fs)
            
    def sim_out(self,delay,decay):
        self.design_oscillation(decay=0,center_freq=0)
        self.y = np.zeros((self.inp_sig.shape[0] + 1,))
        self.run()
        y = self.y#  + 0.01*np.random.normal(1,100,size=self.y.shape)
        
        return y

    def simulate(self,insig=[]):
        
        if insig == []:
            self.design_oscillation(decay=0,center_freq=130,amp_decay=0,amplitude=10)
        elif insig == 'empty':
            self.design_oscillation(decay=0,center_freq=0,amplitude=0,flatline=False)
        else:
            self.inp_sig = insig
        
        #Container for y
        self.y = np.zeros((self.inp_sig.shape[0] + 1,))
            
        self.run()
        
        tvect = self.tvect
        inp_sig = self.inp_sig
        y = self.y #+ 0.009*np.random.normal(0,1,size=self.y.shape)
        fs = self.fs
        
        #LPF here
        sos_lpf = sig.butter(3,100,fs=self.fs,output='sos')
        y = sig.sosfilt(sos_lpf,y)
        
        # #Plotting
        # plt.figure()
        # plt.plot(tvect,inp_sig)
        # plt.plot(tvect,y[:inp_sig.shape[0]],'r')
        
        plt.figure()
        f,pxx = sig.welch(inp_sig,fs=fs,window='blackmanharris',nperseg=512,noverlap=500)
        f,pyy = sig.welch(y,fs=fs,window='blackmanharris',nperseg=512,noverlap=500)
        plt.plot(f,np.log10(pxx))
        plt.plot(f,np.log10(pyy),'r')
        plt.figure()
        plt.loglog(f,pyy)

        plt.figure()
        nseg = 1024
        t,f,sgx = sig.spectrogram(inp_sig,fs=fs,window='blackmanharris',nperseg=nseg,noverlap=nseg-100,nfft=2**10)
        t,f,sgy = sig.spectrogram(y,fs=fs,window='blackmanharris',nperseg=nseg,noverlap=nseg-100,nfft=2**10)
        plt.pcolormesh(f,t,np.log10(sgy))
        plt.ylim((0,30))
        plt.suptitle('Delay samples:'+str(self.k[0]) + ' decay: ' + str(self.k_decay))
        plt.show()


# WORKS NICE! delayer = delay_filt(sec_delay=0.1,k_decay=-0.1)
#flip k_decay for gold: delayer = delay_filt(sec_delay=0.07,k_decay=.-1)
delayer = delay_filt(sec_delay=0.1,k_decay=-.05)
#delayer.design_oscillation()
delayer.simulate(insig=[])