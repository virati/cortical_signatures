#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 13:10:17 2020

@author: virati
"""

import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict
import scipy.signal as sig
import matplotlib
import sys

import matplotlib.pyplot as plt

sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/archive/MMDBS/')
import TimeSeries as ts

from scipy.interpolate import interp1d

import pdb
import matplotlib.colors as colors

from sklearn.decomposition import PCA

import scipy.stats as stats

import sys
sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/')
import DBSpace as dbo
from DBSpace import nestdict

import pickle

#%%
def plot_phase(mI,pI,conditI,chann,SGs,tpts,filt=True,fileio_out=False):
    b,a = sig.butter(10,30/422)
    cm = plt.cm.get_cmap('RdYlBu')
    
    chirp = defaultdict(dict)
    
    for m in mI:
        for p in pI:
            for condit in conditI:
                
                tvec = SGs[m][p][condit]['TRaw']
                sel_tvec = np.logical_and(tvec > tpts[0],tvec < tpts[1])
                
                Lchann = stats.zscore(sig.filtfilt(b,a,SGs[m][p][condit]['Raw'][sel_tvec,0]))
                Rchann = stats.zscore(sig.filtfilt(b,a,SGs[m][p][condit]['Raw'][sel_tvec,1]))
                
                if fileio_out:
                    chirp['Raw'] = SGs[m][p][condit]['Raw'][sel_tvec,:]                    
                    chirp['Filt'] = [Lchann,Rchann]
                
                    pickle.dump(chirp,open('/tmp/test.pickle',"wb"))
                    
                else:
                    plt.figure()
#                    plt.subplot(311)
#                    plt.plot(SGs[m][p][condit]['TRaw'],SGs[m][p][condit]['Raw'])
#                    plt.axis('off')
                    plt.subplot(312)
                    #filter the two
                    plt.plot(tvec[sel_tvec],Lchann)
                    plt.plot(tvec[sel_tvec],Rchann)
                    
                    plt.subplot(313)
                    #plt.scatter(Lchann,Rchann,c=tvec[sel_tvec],marker='.',cmap=cm,alpha=0.1)
                    plt.xlim((-5,5))
                    plt.ylim((-5,5))
                    plt.title('Phase Portrait')
                    
                    chirp['Raw'] = [Lchann,Rchann]
    return chirp


#%%
for mm, modal in enumerate(['LFP']):
    for pp, pt in enumerate(['905']):
        SGs[modal][pt] = defaultdict(dict)
        for cc, condit in enumerate(['OnTarget','OffTarget']):
            Data = []
            Data = ts.import_BR(Ephys[modal][pt][condit]['Filename'],snip=(0,0))
            #Data = dbo.load_BR_dict(Ephys[modal][pt][condit]['Filename'],sec_end=0)
            #Compute the TF representation of the above imported data
            F,T,SG,BANDS = Data.compute_tf()
            SG_Dict = dbo.gen_SG(Data.extract_dict(),overlap=False)
            #Fvect = dbo.calc_feats()
            #for iv, interval in enumerate():
          
            [datatv,dataraw] = Data.raw_ts()
            
            SGs[modal][pt][condit]['SG'] = {chann:SG_Dict[chann]['SG'] for chann in ['Left','Right']}
            SGs[modal][pt][condit]['Raw'] = dataraw
            SGs[modal][pt][condit]['TRaw'] = datatv
            SGs[modal][pt][condit]['T'] = SG_Dict['Left']['T']
            #pdb.set_trace()
            SGs[modal][pt][condit]['Bands'] = BANDS
            SGs[modal][pt][condit]['BandMatrix'] = np.zeros((BANDS[0]['Alpha'].shape[0],2,5))
            SGs[modal][pt][condit]['BandSegments'] = nestdict()
            SGs[modal][pt][condit]['DSV'] = np.zeros((BANDS[0]['Alpha'].shape[0],2,1))
            
            
    SGs[modal]['F'] = SG_Dict['F']

#%%
disp_modal=['LFP']
disp_pt = ['905']
disp_condit = ['OnTarget']
#for 906 bilat: timeseg = [611,690]
timeseg = [606,687]
chirp = plot_phase('LFP',pt,'OffTarget','Right',SGs,tpts=timeseg,fileio_out=False)

chirp_templ = chirp['Raw'][1][10:30*422]
#do chirplet transform on the chirp template

#Ignore chirplet transform/analysis and just use the template to search amongst voltage sweep data
vsweep_fname = '/home/virati/MDD_Data/BR/905/Session_2015_09_02_Wednesday/Dbs905_2015_09_02_10_31_14__MR_0.txt'
#load in the vsweep data
vsweepData = ts.import_BR(vsweep_fname,snip=(0,0))
[vstv,vsraw] = vsweepData.raw_ts()

vsweepData.view_tf(channs=np.arange(2))

#go through vsraw and check for chirp_templ

#How many inner products do we need to compute?
chirp_templ = chirp_templ - np.mean(chirp_templ)

n_tot = vsraw.shape[0]
n_templ = chirp_templ.shape[0]
n_sweep = n_tot - n_templ

tl_ip = np.zeros((n_sweep,2))

for tlag in range(0,n_sweep,10):
    print('tlag = ' + str(tlag))
    #mean zero the current
    curr_sig = vsraw[tlag:tlag+n_templ] - np.mean(vsraw[tlag:tlag+n_templ])
    tl_ip[tlag] = np.dot(chirp_templ,curr_sig)

#%%

tvect = SGs['LFP'][pt]['OnTarget']['TRaw']
conv_tvect = np.linspace(0,tvect[-1],tl_ip.shape[0])
plt.figure()
plt.plot(conv_tvect,tl_ip)
plt.legend(['Channel 0','Channel 1'])

plt.figure()
plt.plot(chirp_templ)


#%%

#file_writeout_the raw ts, the filtered ts, of left and right
#write_chirp(disp_modal,disp_pt,disp_condit,1,SGs,tpts=timeseg)

#%%
          
#Do the segmentation
#What times are most important?