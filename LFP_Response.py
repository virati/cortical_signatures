#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 22:05:44 2018

@author: virati
LFP Response Script
"""

import sys
sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/')
import DBS_Osc as dbo
from DBS_Osc import nestdict

win_list = ['Bilat','PreBilat']

#%%
#File and meta info
class sweep_lfp:
    
    def __init__(self,sweep_type='Targeting'):
        self.load_meta_data()
        self.set_profile(sweep_type)
        self.load_LFP()
        
    def set_profile(self,sweep_type):
        self.sweep_type = sweep_type
        #Anything else we need to do here
        
        if sweep_type == 'Targeting':
            self.condits = ['OnTarget','OffTarget']
            self.pts = ['901','903','905','906','907','908']
        
    def load_LFP(self):
        base_element = self.meta_data['LFP']
        
        if self.sweep_type == 'Targeting':
            LFP_List = {pt:{condit:dbo.load_BR_dict(base_element[pt][condit]['Filename'],sec_end=0) for condit in self.condits} for pt in self.pts}
            
        self.LFP_Data = LFP_List
        
    def impose_frames(self):
        #this function will impose a "topology" of frames, with the basic sets being the null and the full recording
        window_list = {pt:{win_name:{condit:self.meta_data['LFP'][pt][condit]['segments'][win_name] for condit in self.condits} for win_name in win_list} for pt in self.pts}
        self.frames = window_list
        
    def load_meta_data(self):
        Ephys = nestdict()
        Ephys['LFP']['901']['OnTarget']['Filename'] = '/home/virati/MDD_Data/BR/901/Session_2014_05_16_Friday/DBS901_2014_05_16_17_10_31__MR_0.txt'
        Ephys['LFP']['901']['OffTarget']['Filename'] = '/home/virati/MDD_Data/BR/901/Session_2014_05_16_Friday/DBS901_2014_05_16_16_25_07__MR_0.txt'
        Ephys['LFP']['901']['OnTarget']['segments']['Bilat'] = (600,630)
        Ephys['LFP']['901']['OnTarget']['segments']['PreBilat'] = (500,530)
        Ephys['LFP']['901']['OffTarget']['segments']['Bilat'] = (600,630)
        Ephys['LFP']['901']['OffTarget']['segments']['PreBilat'] = (480,510)
        
        Ephys['LFP']['903']['OnTarget']['Filename'] = '/home/virati/MDD_Data/BR/903/Session_2014_09_03_Wednesday/DBS903_2014_09_03_14_16_57__MR_0.txt'
        Ephys['LFP']['903']['OffTarget']['Filename'] = '/home/virati/MDD_Data/BR/903/Session_2014_09_04_Thursday/DBS903_2014_09_04_12_53_09__MR_0.txt' 
        Ephys['LFP']['903']['OnTarget']['segments']['Bilat'] = (550,580)
        Ephys['LFP']['903']['OffTarget']['segments']['Bilat'] = (550,580)
        Ephys['LFP']['903']['OnTarget']['segments']['PreBilat'] = (501,531)
        Ephys['LFP']['903']['OffTarget']['segments']['PreBilat'] = (501,531)
        
        
        
        Ephys['LFP']['905']['OnTarget']['Filename'] = '/home/virati/MDD_Data/BR/905/Session_2015_09_28_Monday/Dbs905_2015_09_28_13_51_42__MR_0.txt' 
        Ephys['LFP']['905']['OffTarget']['Filename'] = '/home/virati/MDD_Data/BR/905/Session_2015_09_29_Tuesday/Dbs905_2015_09_29_12_32_47__MR_0.txt' 
        Ephys['LFP']['905']['OnTarget']['segments']['Bilat'] = (610,640)
        Ephys['LFP']['905']['OffTarget']['segments']['Bilat'] = (610,640)
        Ephys['LFP']['905']['OnTarget']['segments']['PreBilat'] = (561,591)
        Ephys['LFP']['905']['OffTarget']['segments']['PreBilat'] = (561,591)
        
        
        Ephys['LFP']['906']['OnTarget']['Filename'] = '/home/virati/MDD_Data/BR/906/Session_2015_08_27_Thursday/DBS906_2015_08_27_15_10_44__MR_0.txt'
        Ephys['LFP']['906']['OffTarget']['Filename'] = '/home/virati/MDD_Data/BR/906/Session_2015_08_27_Thursday/DBS906_2015_08_27_16_20_23__MR_0.txt'
        Ephys['LFP']['906']['OnTarget']['segments']['Bilat'] = (610,640)
        Ephys['LFP']['906']['OffTarget']['segments']['Bilat'] = (610,640)
        Ephys['LFP']['906']['OnTarget']['segments']['PreBilat'] = (561,591)
        Ephys['LFP']['906']['OffTarget']['segments']['PreBilat'] = (561,591)
        
        
        Ephys['LFP']['907']['OnTarget']['Filename'] = '/home/virati/MDD_Data/BR/907/Session_2015_12_16_Wednesday/DBS907_2015_12_16_12_09_04__MR_0.txt'
        Ephys['LFP']['907']['OffTarget']['Filename'] = '/home/virati/MDD_Data/BR/907/Session_2015_12_17_Thursday/DBS907_2015_12_17_10_53_08__MR_0.txt' 
        Ephys['LFP']['907']['OnTarget']['segments']['Bilat'] = (640,670)
        Ephys['LFP']['907']['OffTarget']['segments']['Bilat'] = (625,655)
        Ephys['LFP']['907']['OnTarget']['segments']['PreBilat'] = (590,620)
        Ephys['LFP']['907']['OffTarget']['segments']['PreBilat'] = (560,590)
        
        
        
        Ephys['LFP']['908']['OnTarget']['Filename'] = '/home/virati/MDD_Data/BR/908/Session_2016_02_10_Wednesday/DBS908_2016_02_10_13_03_10__MR_0.txt'
        Ephys['LFP']['908']['OffTarget']['Filename'] = '/home/virati/MDD_Data/BR/908/Session_2016_02_11_Thursday/DBS908_2016_02_11_12_34_21__MR_0.txt'
        Ephys['LFP']['908']['OnTarget']['segments']['Bilat'] = (611,641)
        Ephys['LFP']['908']['OffTarget']['segments']['Bilat'] = (611,641)
        Ephys['LFP']['908']['OnTarget']['segments']['PreBilat'] = (551,581)
        Ephys['LFP']['908']['OffTarget']['segments']['PreBilat'] = (551,581)

        self.meta_data = Ephys
        

    def plot_tdom(self,window='Bilat',do_pt=[]):
        if do_pt == []:
            do_pt = self.pts
            
        plt.figure()
        for pt in do_pt:
            plt.plot()
            
    def time_to_idx(self,rec,twin):
        
        



#%%
TResp = sweep_lfp()
TResp.load_LFP()
TResp.impose_frames()


# The end structure we want
