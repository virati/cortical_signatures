#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 11:39:33 2021

@author: virati

File to generate PSDs from early part of DOs
"""

import sys
sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/')
import DBSpace as dbo
from DBSpace import nestdict

from matplotlib import cm
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.signal as sig


def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]#3d plotting fun
from mayavi import mlab
from matplotlib import cm
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

#3d plotting fun
from mayavi import mlab

import numpy as np
import scipy.ndimage as ndimage

#%%
Ephys = nestdict()
Phase = 'TurnOn'
if Phase == 'TurnOn':
    Ephys['901']['OnTarget']['Filename'] = '/home/virati/MDD_Data/BR/901/Session_2014_05_16_Friday/DBS901_2014_05_16_17_10_31__MR_0.txt'
    Ephys['901']['OffTarget']['Filename'] = '/home/virati/MDD_Data/BR/901/Session_2014_05_16_Friday/DBS901_2014_05_16_16_25_07__MR_0.txt'
    Ephys['901']['OnTarget']['segments']['Bilat'] = (600,630)
    Ephys['901']['OnTarget']['segments']['PreBilat'] = (500,530)
    Ephys['901']['OffTarget']['segments']['Bilat'] = (600,630)
    Ephys['901']['OffTarget']['segments']['PreBilat'] = (480,510)
    
    
    Ephys['901']['OffTarget']['segments']['C1'] = (Ephys['901']['OffTarget']['segments']['Bilat'][0],Ephys['901']['OffTarget']['segments']['Bilat'][0] + 15)
    Ephys['901']['OffTarget']['segments']['C2'] = (615,620)
    Ephys['901']['OffTarget']['segments']['C3'] = (620,627)
    Ephys['901']['OffTarget']['segments']['C4'] = (627,635)
    Ephys['901']['OffTarget']['segments']['C5'] = (635,675)    
    
    Ephys['903']['OnTarget']['Filename'] = '/home/virati/MDD_Data/BR/903/Session_2014_09_03_Wednesday/DBS903_2014_09_03_14_16_57__MR_0.txt'
    Ephys['903']['OffTarget']['Filename'] = '/home/virati/MDD_Data/BR/903/Session_2014_09_04_Thursday/DBS903_2014_09_04_12_53_09__MR_0.txt' 
    Ephys['903']['OnTarget']['segments']['Bilat'] = (550,580)
    Ephys['903']['OffTarget']['segments']['Bilat'] = (550,580)
    Ephys['903']['OnTarget']['segments']['PreBilat'] = (501,531)
    Ephys['903']['OffTarget']['segments']['PreBilat'] = (501,531)
    
    
    Ephys['903']['OffTarget']['segments']['C1'] = (Ephys['903']['OffTarget']['segments']['Bilat'][0], Ephys['903']['OffTarget']['segments']['Bilat'][0]+15)
    Ephys['903']['OffTarget']['segments']['C2'] = (615,620)
    Ephys['903']['OffTarget']['segments']['C3'] = (620,627)
    Ephys['903']['OffTarget']['segments']['C4'] = (627,635)
    Ephys['903']['OffTarget']['segments']['C5'] = (635,675)    
    
    Ephys['905']['OnTarget']['Filename'] = '/home/virati/MDD_Data/BR/905/Session_2015_09_28_Monday/Dbs905_2015_09_28_13_51_42__MR_0.txt' 
    Ephys['905']['OffTarget']['Filename'] = '/home/virati/MDD_Data/BR/905/Session_2015_09_29_Tuesday/Dbs905_2015_09_29_12_32_47__MR_0.txt' 
    Ephys['905']['OnTarget']['segments']['Bilat'] = (610,640)
    Ephys['905']['OffTarget']['segments']['Bilat'] = (610,640)
    Ephys['905']['OnTarget']['segments']['PreBilat'] = (561,591)
    Ephys['905']['OffTarget']['segments']['PreBilat'] = (561,591)
    
    Ephys['905']['OffTarget']['segments']['C1'] = (Ephys['905']['OffTarget']['segments']['Bilat'][0], Ephys['905']['OffTarget']['segments']['Bilat'][0] + 15)
    Ephys['905']['OffTarget']['segments']['C2'] = (615,620)
    Ephys['905']['OffTarget']['segments']['C4'] = (627,635)
    Ephys['905']['OffTarget']['segments']['C5'] = (635,675)    
    Ephys['905']['OffTarget']['segments']['C3'] = (620,627)
    
    Ephys['906']['OnTarget']['Filename'] = '/home/virati/MDD_Data/BR/906/Session_2015_08_27_Thursday/DBS906_2015_08_27_15_10_44__MR_0.txt'
    Ephys['906']['OffTarget']['Filename'] = '/home/virati/MDD_Data/BR/906/Session_2015_08_27_Thursday/DBS906_2015_08_27_16_20_23__MR_0.txt'
    Ephys['906']['OnTarget']['segments']['Bilat'] = (610,640)
    Ephys['906']['OffTarget']['segments']['Bilat'] = (610,640)
    Ephys['906']['OnTarget']['segments']['PreBilat'] = (561,591)
    Ephys['906']['OffTarget']['segments']['PreBilat'] = (561,591)
    
    #for bilat
    Ephys['906']['OffTarget']['segments']['C1'] = (Ephys['903']['OffTarget']['segments']['Bilat'][0], Ephys['903']['OffTarget']['segments']['Bilat'][0] + 15)
    Ephys['906']['OffTarget']['segments']['C2'] = (615,620)
    Ephys['906']['OffTarget']['segments']['C3'] = (620,627)
    Ephys['906']['OffTarget']['segments']['C4'] = (627,635)
    Ephys['906']['OffTarget']['segments']['C5'] = (635,675)    
    
    Ephys['907']['OnTarget']['Filename'] = '/home/virati/MDD_Data/BR/907/Session_2015_12_16_Wednesday/DBS907_2015_12_16_12_09_04__MR_0.txt'
    Ephys['907']['OffTarget']['Filename'] = '/home/virati/MDD_Data/BR/907/Session_2015_12_17_Thursday/DBS907_2015_12_17_10_53_08__MR_0.txt' 
    Ephys['907']['OnTarget']['segments']['Bilat'] = (640,670)
    Ephys['907']['OffTarget']['segments']['Bilat'] = (625,655)
    Ephys['907']['OnTarget']['segments']['PreBilat'] = (590,620)
    Ephys['907']['OffTarget']['segments']['PreBilat'] = (560,590)
    
    #for bilat
    Ephys['907']['OnTarget']['segments']['C1'] = (Ephys['907']['OnTarget']['segments']['Bilat'][0], Ephys['907']['OnTarget']['segments']['Bilat'][0] + 15)
    
    
    Ephys['908']['OnTarget']['Filename'] = '/home/virati/MDD_Data/BR/908/Session_2016_02_10_Wednesday/DBS908_2016_02_10_13_03_10__MR_0.txt'
    Ephys['908']['OffTarget']['Filename'] = '/home/virati/MDD_Data/BR/908/Session_2016_02_11_Thursday/DBS908_2016_02_11_12_34_21__MR_0.txt'
    Ephys['908']['OnTarget']['segments']['Bilat'] = (611,641)
    Ephys['908']['OffTarget']['segments']['Bilat'] = (611,641)
    Ephys['908']['OnTarget']['segments']['PreBilat'] = (551,581)
    Ephys['908']['OffTarget']['segments']['PreBilat'] = (551,581)
    
    
    Ephys['908']['OnTarget']['segments']['C1'] = (Ephys['908']['OnTarget']['segments']['Bilat'][0], Ephys['908']['OnTarget']['segments']['Bilat'][0] + 15)
    Ephys['908']['OnTarget']['segments']['C2'] = (615,620)
    
elif Phase == '6Mo':
            #901
    Ephys['901']['OnTarget']['Filename'] = '/run/media/virati/Samsung USB/MDD_Data/BR/901/Session_2014_11_14_Friday/DBS901_2014_11_14_16_46_35__MR_0.txt'
    Ephys['901']['OffTarget']['Filename'] = '/run/media/virati/Samsung USB/MDD_Data/BR/901/Session_2014_11_14_Friday/DBS901_2014_11_14_17_34_35__MR_0.txt'
    Ephys['901']['OnTarget']['segments']['Bilat'] = (670,700)
    Ephys['901']['OnTarget']['segments']['PreBilat'] = (620,650)
    
    Ephys['901']['OffTarget']['segments']['Bilat'] = ()
    Ephys['901']['OffTarget']['segments']['PreBilat'] = ()
    
            #903
    Ephys['903']['OnTarget']['Filename'] = ''
    Ephys['903']['OffTarget']['Filename'] = ''
    
    Ephys['903']['OnTarget']['segments']['PreBilat'] = ()
    Ephys['903']['OnTarget']['segments']['Bilat'] = ()
    Ephys['903']['OffTarget']['segments']['PreBilat'] = ()
    Ephys['903']['OffTarget']['segments']['Bilat'] = ()
    
            #905
    Ephys['905']['OnTarget']['Filename'] = ''
    Ephys['905']['OffTarget']['Filename'] = ''
    Ephys['905']['OnTarget']['segments']['PreBilat'] = ()
    Ephys['905']['OnTarget']['segments']['Bilat'] = ()
    Ephys['905']['OffTarget']['segments']['PreBilat'] = ()
    Ephys['905']['OffTarget']['segments']['Bilat'] = ()
    
            #906
    Ephys['906']['OnTarget']['Filename'] = ''
    Ephys['906']['OffTarget']['Filename'] = ''
    Ephys['906']['OnTarget']['segments']['Bilat'] = (610,640)
    Ephys['906']['OffTarget']['segments']['Bilat'] = (610,640)
    Ephys['906']['OnTarget']['segments']['PreBilat'] = (561,591)
    Ephys['906']['OffTarget']['segments']['PreBilat'] = (561,591)
    
            #907
    Ephys['907']['OnTarget']['Filename'] = ''
    Ephys['907']['OffTarget']['Filename'] = ''
    Ephys['907']['OnTarget']['segments']['Bilat'] = (640,670)
    Ephys['907']['OffTarget']['segments']['Bilat'] = (625,655)
    Ephys['907']['OnTarget']['segments']['PreBilat'] = (590,620)
    Ephys['907']['OffTarget']['segments']['PreBilat'] = (560,590)
    
            #908
    Ephys['908']['OnTarget']['Filename'] = ''
    Ephys['908']['OffTarget']['Filename'] = ''
    Ephys['908']['OnTarget']['segments']['Bilat'] = (611,641)
    Ephys['908']['OffTarget']['segments']['Bilat'] = (611,641)
    Ephys['908']['OnTarget']['segments']['PreBilat'] = (551,581)
    Ephys['908']['OffTarget']['segments']['PreBilat'] = (551,581)
    
do_presence = {'901':('OffTarget','Left'),'903':('OffTarget','Left'),'905':('OffTarget','Left'),'906':('OffTarget','Right'),'907':('OnTarget','Left'),'908':('OnTarget','Left')}

SGs = nestdict()
#%%
pt_list = ['901','903','905','906','907','908']
TS = nestdict()
for pp, pt in enumerate(pt_list):
    for cc, condit in enumerate(['OnTarget','OffTarget']):
        Data_In = dbo.load_BR_dict(Ephys[pt][condit]['Filename'],sec_offset=0,lpf=20)
        
        TS[pt][condit] = Data_In
        SGs[pt][condit] = dbo.gen_SG(Data_In)
        #CWTs[pt][condit] = dbo.gen_CWT(Data_In)

#%%
#Here we'll zoom into the details of the 906_OFFT DO
time_zoom = {'906':(547,800),
             '901':None,
             '903':None,
             '905':None,
             '907': None,
             '908':None}

for pt in ['906']:
    side=do_presence[pt][1]
    condit = do_presence[pt][0]
    tvect = np.linspace(0,len(TS[pt][condit][side]) / 422, len(TS[pt][condit][side]))
    
    plt.figure()
    
    plt.plot(tvect,TS[pt][condit][side])
    if time_zoom[pt] is not None:
        pass
        plt.xlim(time_zoom[pt])
        
    
    plt.figure()
    tvect = SGs[pt][condit][side]['T']
    Fvect = SGs[pt][condit][side]['F']
    SGdata = SGs[pt][condit][side]['SG']
    
    plt.pcolormesh(tvect,Fvect,10*np.log10(SGdata),rasterized=True)
    if time_zoom[pt] is not None:
        plt.xlim(time_zoom[pt])
    
        x_idxs = np.where(np.logical_and(tvect > time_zoom[pt][0],tvect < time_zoom[pt][1]))
    
    plt.title(f"{pt} at {condit} with recording {side}")
    plt.colorbar()
    
    
#%%
#Below is obviously broken for non-906 since the segment 'C's aren't defined
plt.figure()
pt_colors = ['r','b','g','k','m','c']
for pp,pt in enumerate(pt_list):
    do_condit = do_presence[pt][0]
    do_rec_side = do_presence[pt][1]
    stim_side = 'Bilat'
    
    print(f"{pt} with {condit}")
    seg = 'C1'
    t_vect = SGs[pt][condit][do_rec_side]['T']
    f_vect = SGs[pt][condit][do_rec_side]['F']
    sg_data = SGs[pt][do_condit][do_rec_side]['SG']
    
    #find indices for times
    baseline_start_idx = Ephys[pt][do_condit]['segments']['PreBilat'][0]
    baseline_end_idx = Ephys[pt][do_condit]['segments']['PreBilat'][1]
    baseline_time_idxs = np.where(np.logical_and(t_vect > baseline_start_idx, t_vect < baseline_end_idx))
    mean_baseline_data = sg_data[:,baseline_time_idxs].squeeze().mean(axis=1)
    
    start_idx = Ephys[pt][do_condit]['segments'][seg][0]
    end_idx = Ephys[pt][do_condit]['segments'][seg][1]
    time_idxs = np.where(np.logical_and(t_vect > start_idx, t_vect < end_idx))
    mean_data = sg_data[:,time_idxs].squeeze().mean(axis=1)
    std_data = (np.std(sg_data[:,time_idxs].squeeze(),axis=1))
    
    mean_psd = mean_data/mean_baseline_data
    mean_curve = 10*np.log10(mean_psd)
    mean_top = 10*np.log10(mean_psd + std_data)
    mean_bottom = 10*np.log10(mean_psd - std_data)
    
    plt.plot(f_vect,mean_curve,label=pt, color=pt_colors[pp])
    #plt.fill_between(f_vect, mean_bottom, mean_top, color=pt_colors[pp],alpha=0.2)
    
    #plt.xlim((0,32))
plt.legend()


#%%
plt.figure()
for pp,pt in enumerate(['906']):
    do_condit = do_presence[pt][0]
    do_rec_side = do_presence[pt][1]
    stim_side = 'Bilat'
    
    print(f"{pt} with {condit}")
    seg = 'C1'
    t_vect = SGs[pt][condit][do_rec_side]['T']
    f_vect = SGs[pt][condit][do_rec_side]['F']
    sg_data = SGs[pt][do_condit][do_rec_side]['SG']
    
    #find indices for times
    baseline_start_idx = Ephys[pt][do_condit]['segments']['PreBilat'][0]
    baseline_end_idx = Ephys[pt][do_condit]['segments']['PreBilat'][1]
    baseline_time_idxs = np.where(np.logical_and(t_vect > baseline_start_idx, t_vect < baseline_end_idx))
    mean_baseline_data = sg_data[:,baseline_time_idxs].squeeze().mean(axis=1)
    
    for ee in range(1,10):
        start_idx = Ephys[pt][do_condit]['segments'][seg][0] + 20 * (ee-1)
        end_idx = Ephys[pt][do_condit]['segments'][seg][0] + 20 * ee
        time_idxs = np.where(np.logical_and(t_vect > start_idx, t_vect < end_idx))
        mean_data = sg_data[:,time_idxs].squeeze().mean(axis=1)
        
        plt.plot(f_vect,10*np.log10(mean_data/mean_baseline_data),label=f"{ee * 20} window",alpha=0.8,linewidth=5)
    plt.xlim((0,32))
    plt.ylim((-11,66))
    plt.title(f"{pt} PSD Deviation from Pre-Stimulation")
plt.legend()
