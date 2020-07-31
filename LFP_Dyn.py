#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 23:17:47 2018

@author: virati
LFP Dynamics script
Captures chirp changes
"""

import sys
sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/')
import DBSpace as dbo
from DBSpace import nestdict

import matplotlib.pyplot as plt
import numpy as np
import math


def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]
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
    
    Ephys['903']['OnTarget']['Filename'] = '/home/virati/MDD_Data/BR/903/Session_2014_09_03_Wednesday/DBS903_2014_09_03_14_16_57__MR_0.txt'
    Ephys['903']['OffTarget']['Filename'] = '/home/virati/MDD_Data/BR/903/Session_2014_09_04_Thursday/DBS903_2014_09_04_12_53_09__MR_0.txt' 
    Ephys['903']['OnTarget']['segments']['Bilat'] = (550,580)
    Ephys['903']['OffTarget']['segments']['Bilat'] = (550,580)
    Ephys['903']['OnTarget']['segments']['PreBilat'] = (501,531)
    Ephys['903']['OffTarget']['segments']['PreBilat'] = (501,531)
    
    
    
    Ephys['905']['OnTarget']['Filename'] = '/home/virati/MDD_Data/BR/905/Session_2015_09_28_Monday/Dbs905_2015_09_28_13_51_42__MR_0.txt' 
    Ephys['905']['OffTarget']['Filename'] = '/home/virati/MDD_Data/BR/905/Session_2015_09_29_Tuesday/Dbs905_2015_09_29_12_32_47__MR_0.txt' 
    Ephys['905']['OnTarget']['segments']['Bilat'] = (610,640)
    Ephys['905']['OffTarget']['segments']['Bilat'] = (610,640)
    Ephys['905']['OnTarget']['segments']['PreBilat'] = (561,591)
    Ephys['905']['OffTarget']['segments']['PreBilat'] = (561,591)
    
    
    Ephys['906']['OnTarget']['Filename'] = '/home/virati/MDD_Data/BR/906/Session_2015_08_27_Thursday/DBS906_2015_08_27_15_10_44__MR_0.txt'
    Ephys['906']['OffTarget']['Filename'] = '/home/virati/MDD_Data/BR/906/Session_2015_08_27_Thursday/DBS906_2015_08_27_16_20_23__MR_0.txt'
    Ephys['906']['OnTarget']['segments']['Bilat'] = (610,640)
    Ephys['906']['OffTarget']['segments']['Bilat'] = (610,640)
    Ephys['906']['OnTarget']['segments']['PreBilat'] = (561,591)
    Ephys['906']['OffTarget']['segments']['PreBilat'] = (561,591)
    
    Ephys['906']['OffTarget']['segments']['C1'] = (368,389)
    Ephys['906']['OffTarget']['segments']['C2'] = (389,422)
    Ephys['906']['OffTarget']['segments']['C3'] = (422,475)
    Ephys['906']['OffTarget']['segments']['C4'] = (475,486)
    Ephys['906']['OffTarget']['segments']['C5'] = (488,530)
    
    
    Ephys['907']['OnTarget']['Filename'] = '/home/virati/MDD_Data/BR/907/Session_2015_12_16_Wednesday/DBS907_2015_12_16_12_09_04__MR_0.txt'
    Ephys['907']['OffTarget']['Filename'] = '/home/virati/MDD_Data/BR/907/Session_2015_12_17_Thursday/DBS907_2015_12_17_10_53_08__MR_0.txt' 
    Ephys['907']['OnTarget']['segments']['Bilat'] = (640,670)
    Ephys['907']['OffTarget']['segments']['Bilat'] = (625,655)
    Ephys['907']['OnTarget']['segments']['PreBilat'] = (590,620)
    Ephys['907']['OffTarget']['segments']['PreBilat'] = (560,590)
    
    
    
    Ephys['908']['OnTarget']['Filename'] = '/home/virati/MDD_Data/BR/908/Session_2016_02_10_Wednesday/DBS908_2016_02_10_13_03_10__MR_0.txt'
    Ephys['908']['OffTarget']['Filename'] = '/home/virati/MDD_Data/BR/908/Session_2016_02_11_Thursday/DBS908_2016_02_11_12_34_21__MR_0.txt'
    Ephys['908']['OnTarget']['segments']['Bilat'] = (611,641)
    Ephys['908']['OffTarget']['segments']['Bilat'] = (611,641)
    Ephys['908']['OnTarget']['segments']['PreBilat'] = (551,581)
    Ephys['908']['OffTarget']['segments']['PreBilat'] = (551,581)
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
    

SGs = nestdict()
#%%
for pp, pt in enumerate(['906']):
    for cc, condit in enumerate(['OnTarget','OffTarget']):
        Data_In = dbo.load_BR_dict(Ephys[pt][condit]['Filename'],sec_offset=0)
        
        SGs[pt][condit] = dbo.gen_SG(Data_In)
   
for pp,pt in enumerate(['906']):
    plt.figure()
    for cc, condit in enumerate(['OffTarget']):
        do_segs = ['C1','C2','C3','C4']
        for seg in do_segs:
            #find indices for times
            start_idx = min(range(SGs[pt][condit]['Left']['T'].shape[0]), key=lambda i: abs(SGs[pt][condit]['Left']['T'][i]-Ephys[pt][condit]['segments'][seg][0]))
            end_idx = min(range(SGs[pt][condit]['Left']['T'].shape[0]), key=lambda i: abs(SGs[pt][condit]['Left']['T'][i]-Ephys[pt][condit]['segments'][seg][1]))
            
            
            
            plt.plot(SGs[pt][condit]['Left']['F'],10*np.log10(np.median(SGs[pt][condit]['Left']['SG'][:,start_idx:end_idx],axis=1)))
        plt.legend(do_segs)
#%%
for pp, pt in enumerate(['906']):
    plt.figure()
    plt.suptitle(pt)
    for cc, condit in enumerate(['OnTarget','OffTarget']):
        plt.subplot(2,2,2*cc+1)
        plt.title(condit)
        plt.pcolormesh(SGs[pt][condit]['Left']['T'],SGs[pt][condit]['Left']['F'],10*np.log10(SGs[pt][condit]['Left']['SG']),rasterized=True)
        plt.subplot(2,2,2*cc+2)
        plt.title(condit)
        plt.pcolormesh(SGs[pt][condit]['Right']['T'],SGs[pt][condit]['Right']['F'],10*np.log10(SGs[pt][condit]['Right']['SG']),rasterized=True)
        #%%
