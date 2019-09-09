# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 15:19:36 2016

@author: virati
This is now the actual code for doing OnTarget/OffTarget LFP Ephys
"""

#Manifold Learning Preprocessing

import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict
import scipy.signal as sig
import matplotlib
import sys

import matplotlib.pyplot as plt

sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/MMDBS/')
import TimeSeries as ts

from scipy.interpolate import interp1d

import pdb
import matplotlib.colors as colors

from sklearn.decomposition import PCA

import scipy.stats as stats

import sys
sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/')
import DBS_Osc as dbo

import pickle

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}

matplotlib.rc('font', **font)
matplotlib.rcParams['svg.fonttype'] = 'none'

import seaborn as sns

sns.set_context('talk')
sns.set_style('white')
sns.set_style('ticks')
sns.set(font_scale=3)

plt.rcParams['image.cmap'] = 'jet'
#%%
#Data.view_raw_ts(channs=range(2))
#Data.view_raw_hist(chann=[0,1])
def plot_SG(mI,pI,conditI,chann,SGs,tpts=0):
            
    plt.figure()
    for m in mI:
        for p in pI:
            for condit in conditI:
                plt.figure()
                if tpts != 0:
                    t_idxs = np.where(np.logical_and(SGs[m][p][condit]['T'] > tpts[0],SGs[m][p][condit]['T'] < tpts[1]))
                else:
                    t_idxs = np.arange(SGs[m][p][condit]['T'].shape[0])

                plt.pcolormesh(SGs[m][p][condit]['T'][t_idxs],SGs[m]['F'],10*np.log10(np.squeeze(np.abs(SGs[m][p][condit]['SG'][chann][:,t_idxs]))))
                plt.title(m + ' ' + p + ' ' + condit)
                #plt.axis('off')
    #plt.tight_layout()
    #plt.autoscale(enable=True,tight=True)
    #plt.ylim((0,50))
    plt.xlabel('Time (sec)')
    plt.ylabel('Frequency (Hz)')
    plt.show()
    
def plot_PSDs(mI,pI,conditI,chann,SGs):
    for m in mI:
        for p in pI:
            for condit in conditI:
                plt.figure()
                plt.plot(SGs[m]['F'],10*np.log10(np.squeeze(np.abs(np.median(SGs[m][p][condit]['SG'][chann][:,:],axis=1)))))
                plt.axis('off')
def plot_ts(mI,pI,conditI,chann, SGs,tpts, filt=True):
    
    for m in mI:
        for p in pI:
            for condit in conditI:
                #tvec = SGs[m][p][condit]['TRaw']
                #sel_tvec = np.logical_and(tvec > tpts[0],tvec < tpts[1])
                
                #Lchann = stats.zscore(sig.filtfilt(b,a,SGs[m][p][condit]['Raw'][sel_tvec,0]))
                #Rchann = stats.zscore(sig.filtfilt(b,a,SGs[m][p][condit]['Raw'][sel_tvec,1]))
                
                plt.figure()
                plt.subplot(311)
                plt.plot(SGs[m][p][condit]['TRaw'],SGs[m][p][condit]['Raw'])
                
                plt.subplot(312)
                    
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
                    
                
                
                    

#Get the banded power for each band now
def get_SG_Bands(m,p,condit,SGs,band):
    
    band_vect = np.where(np.logical_and(SGs[m]['F'] > band['range'][0],SGs[m]['F'] < band['range'][1]))
    Band_TC = defaultdict(dict)
    
    for cc in range(2):    
        Band_TC[band['label']] = []
        Band_TC[band['label']].append(np.mean(SGs[m][p][condit][cc]['SG'][band_vect,:],0))
    return Band_TC

def get_SG_Bands(m,p,condit,SGs,band):
    
    band_vect = np.where(np.logical_and(SGs[m]['F'] > band['range'][0],SGs[m]['F'] < band['range'][1]))
    Band_TC = defaultdict(dict)
    
    for cc in range(2):    
        Band_TC[band['label']] = []
        Band_TC[band['label']].append(np.mean(SGs[m][p][condit][cc]['SG'][band_vect,:],0))
    return Band_TC


#%%

plt.close('all')

Ephys = defaultdict(dict)
modalities = ['LFP']
patients = ['901','903','905','906','907','908']
condits = ['OnTarget','OffTarget']

for mod in modalities:
    Ephys[mod] = defaultdict(dict)
    for pt in patients:
        Ephys[mod][pt] = defaultdict(dict)
        for cnd in condits:
            Ephys[mod][pt][cnd] = defaultdict(dict)
    
#%%
#Only for TurnOn for now
Phase = 'TurnOn'
if Phase == 'TurnOn':
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
elif Phase == '6Mo':
            #901
    Ephys['LFP']['901']['OnTarget']['Filename'] = '/run/media/virati/Samsung USB/MDD_Data/BR/901/Session_2014_11_14_Friday/DBS901_2014_11_14_16_46_35__MR_0.txt'
    Ephys['LFP']['901']['OffTarget']['Filename'] = '/run/media/virati/Samsung USB/MDD_Data/BR/901/Session_2014_11_14_Friday/DBS901_2014_11_14_17_34_35__MR_0.txt'
    Ephys['LFP']['901']['OnTarget']['segments']['Bilat'] = (670,700)
    Ephys['LFP']['901']['OnTarget']['segments']['PreBilat'] = (620,650)
    
    Ephys['LFP']['901']['OffTarget']['segments']['Bilat'] = ()
    Ephys['LFP']['901']['OffTarget']['segments']['PreBilat'] = ()
    
            #903
    Ephys['LFP']['903']['OnTarget']['Filename'] = ''
    Ephys['LFP']['903']['OffTarget']['Filename'] = ''
    
    Ephys['LFP']['903']['OnTarget']['segments']['PreBilat'] = ()
    Ephys['LFP']['903']['OnTarget']['segments']['Bilat'] = ()
    Ephys['LFP']['903']['OffTarget']['segments']['PreBilat'] = ()
    Ephys['LFP']['903']['OffTarget']['segments']['Bilat'] = ()
    
            #905
    Ephys['LFP']['905']['OnTarget']['Filename'] = ''
    Ephys['LFP']['905']['OffTarget']['Filename'] = ''
    Ephys['LFP']['905']['OnTarget']['segments']['PreBilat'] = ()
    Ephys['LFP']['905']['OnTarget']['segments']['Bilat'] = ()
    Ephys['LFP']['905']['OffTarget']['segments']['PreBilat'] = ()
    Ephys['LFP']['905']['OffTarget']['segments']['Bilat'] = ()
    
            #906
    Ephys['LFP']['906']['OnTarget']['Filename'] = ''
    Ephys['LFP']['906']['OffTarget']['Filename'] = ''
    Ephys['LFP']['906']['OnTarget']['segments']['Bilat'] = (610,640)
    Ephys['LFP']['906']['OffTarget']['segments']['Bilat'] = (610,640)
    Ephys['LFP']['906']['OnTarget']['segments']['PreBilat'] = (561,591)
    Ephys['LFP']['906']['OffTarget']['segments']['PreBilat'] = (561,591)
    
            #907
    Ephys['LFP']['907']['OnTarget']['Filename'] = ''
    Ephys['LFP']['907']['OffTarget']['Filename'] = ''
    Ephys['LFP']['907']['OnTarget']['segments']['Bilat'] = (640,670)
    Ephys['LFP']['907']['OffTarget']['segments']['Bilat'] = (625,655)
    Ephys['LFP']['907']['OnTarget']['segments']['PreBilat'] = (590,620)
    Ephys['LFP']['907']['OffTarget']['segments']['PreBilat'] = (560,590)
    
            #908
    Ephys['LFP']['908']['OnTarget']['Filename'] = ''
    Ephys['LFP']['908']['OffTarget']['Filename'] = ''
    Ephys['LFP']['908']['OnTarget']['segments']['Bilat'] = (611,641)
    Ephys['LFP']['908']['OffTarget']['segments']['Bilat'] = (611,641)
    Ephys['LFP']['908']['OnTarget']['segments']['PreBilat'] = (551,581)
    Ephys['LFP']['908']['OffTarget']['segments']['PreBilat'] = (551,581)
    
#%%
#import shutil
#
#for mod in modalities:
#    for pt in patients:
#        
#        for cnd in condits:
#            #copy the file to tmp:
#            shutil.copyfile(Ephys[mod][pt][cnd]['Filename'],'/tmp/data_upload/' + pt + '_' + mod + '_' + cnd)
            
#%%
#Load in the files
plt.close('all')

SCC_State = defaultdict(dict)
SGs = defaultdict(dict)

SGs['LFP'] = defaultdict()
#set the Fs for LFPs here
SGs['LFP']['Fs'] = 422

do_DSV = np.array([[-0.00583578, -0.00279751,  0.00131825,  0.01770169,  0.01166687],[-1.06586005e-02,  2.42700023e-05,  7.31445236e-03,  2.68723035e-03,-3.90440108e-06]])

do_DSV = do_DSV / np.linalg.norm(do_DSV)

#Loop through the modalities and import the BR recording
for mm, modal in enumerate(['LFP']):
    for pp, pt in enumerate(['901','903','905','906','907','908']):
        SGs[modal][pt] = defaultdict(dict)
        for cc, condit in enumerate(['OnTarget','OffTarget']):
            Data = []
            Data = ts.import_BR(Ephys[modal][pt][condit]['Filename'],snip=(0,0))
            #Data = dbo.load_BR_dict(Ephys[modal][pt][condit]['Filename'],sec_end=0)
            #Compute the TF representation of the above imported data
            F,T,SG,BANDS = Data.compute_tf()
            SG_Dict = dbo.gen_SG(Data)
            #Fvect = dbo.calc_feats()
            #for iv, interval in enumerate():
          
            #[datatv,dataraw] = Data.raw_ts()
            
            SGs[modal][pt][condit]['SG'] = SG_Dict['SG']
            #SGs[modal][pt][condit]['Raw'] = dataraw
            #SGs[modal][pt][condit]['TRaw'] = datatv
            SGs[modal][pt][condit]['T'] = SG_Dict['T']
            SGs[modal][pt][condit]['Bands'] = BANDS
            SGs[modal][pt][condit]['BandMatrix'] = np.zeros((BANDS[0]['Alpha'].shape[0],2,5))
            #SGs[modal][pt][condit]['BandSegments'] = []
            #SGs[modal][pt][condit]['DSV'] = np.zeros((BANDS[0]['Alpha'].shape[0],2,1))
            
            
    SGs[modal]['F'] = SG_Dict['F']
    
#%%
#Segment the data based on prescribed segmentations
#bands = ts.band_structs()
do_bands = dbo.feat_order

Response_matrix = np.zeros((6,2,2,5))


for mm, modal in enumerate(['LFP']):
    for pp, pt in enumerate(['901','903','905','906','907','908']):
        for co, condit in enumerate(['OnTarget','OffTarget']):
            SGs[modal][pt][condit]['BandSegments'] = defaultdict(dict)
            for bb, bands in enumerate(do_bands):
                for cc in range(2):
                    SGs[modal][pt][condit]['BandMatrix'][:,cc,bb] = SGs[modal][pt][condit]['Bands'][cc][bands]
                    for seg in range(SGs[modal][pt][condit]['BandMatrix'].shape[0]):
                        SGs[modal][pt][condit]['DSV'][seg,cc] = np.dot(SGs[modal][pt][condit]['BandMatrix'][seg,cc,:],do_DSV[cc,:])
                        
            for sg, seg in enumerate(Ephys[modal][pt][condit]['segments'].keys()):
                tbounds = [Ephys[modal][pt][condit]['segments'][seg][0],Ephys[modal][pt][condit]['segments'][seg][1]]
                #extract from time vector the actual indices
                t_idxs = np.ceil(np.where(np.logical_and(SGs[modal][pt][condit]['T'] >= tbounds[0],SGs[modal][pt][condit]['T'] <= tbounds[1]))).astype(int)
                SGs[modal][pt][condit]['BandSegments'][seg]=SGs[modal][pt][condit]['BandMatrix'][t_idxs,:,:]
                SGs[modal][pt][condit][seg] = defaultdict(dict)
                SGs[modal][pt][condit][seg]['PCA'] = defaultdict(dict)
            #This is a (2,4) matrix, with channel x band
            SGs[modal][pt][condit]['Response'] = 10* np.log10(np.mean(SGs[modal][pt][condit]['BandSegments']['Bilat'][0,:,:,:],0)) - 10* np.log10(np.mean(SGs[modal][pt][condit]['BandSegments']['PreBilat'][0,:,:,:],0))
            Response_matrix[pp,co,:,:] = SGs[modal][pt][condit]['Response']


#%%
#Plot DSV directions/control theory
for pt in ['901','903','905','906','907','908']:
    plt.figure()
    for cc,condit in enumerate(['OnTarget','OffTarget']):
        plt.subplot(2,1,cc+1)
        #plt.plot(SGs[modal][pt][condit]['DSV'][:,0],label='Left LFP')
        #plt.plot(SGs[modal][pt][condit]['DSV'][:,1],label='Right LFP')
        
        llfp = stats.zscore(SGs[modal][pt][condit]['DSV'][:,0]).squeeze()
        rlfp = stats.zscore(SGs[modal][pt][condit]['DSV'][:,1]).squeeze()
        orig_len=len(llfp)
        ti = np.linspace(2,orig_len+1,10*orig_len)
        
        li = np.concatenate((llfp[-3:-1],llfp,llfp[1:3]))
        ri = np.concatenate((rlfp[-3:-1],rlfp,rlfp[1:3]))
        t = np.arange(li.shape[0])
        
        lii = interp1d(t,li,kind='cubic')(ti)
        rii = interp1d(t,ri,kind='cubic')(ti)
        
        plt.scatter(llfp,rlfp,c=np.linspace(0,1,llfp.shape[0]),cmap='cool')
        plt.plot(lii,rii,alpha=0.2)
        #for ii in range(len(lii)):
            #p = plt.plot(lii,rii,color=pl.cm.jet(np.linspace(0,1,len(lii)))[ii])
        #colorline(lii,rii,np.linspace(0,1,len(lii)),cmap=plt.get_cmap('jet'))
        plt.ylim((-0.8,0.8))
        plt.xlim((-0.8,0.8))
        plt.title(condit)
    plt.suptitle(pt)
#%%
#Now, just plot in a boxplot format what we want
condit = 0
#plt.figure()
ax = plt.subplot(121)
plt.boxplot(Response_matrix[:,condit,0,2])
plt.ylim((-3,30))
plt.axhline(y=0)
plt.xticks([1],['Alpha'])
plt.xlabel('Oscillation')
plt.ylabel('Power Change (dB)')
plt.title('Left LFP Channel')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)


ax = plt.subplot(122)
plt.boxplot(Response_matrix[:,condit,1,2])
plt.ylim((-3,30))
plt.axhline(y=0)
plt.xticks([1],['Alpha'])
plt.xlabel('Oscillation')
plt.ylabel('Power Change (dB)')
plt.title('Right LFP Channel')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)

plt.show()
#%%
#Now, plot both conditions next to each other
#plt.figure()
bb = 2

#for bb,band in enumerate(do_bands):
plt.figure()
ax = plt.subplot(121)
bp = plt.boxplot(Response_matrix[:,0,0,:],positions=np.linspace(1,6,5),widths=0.25)
edge_color='blue'
for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp[element], color=edge_color)
    
bp = plt.boxplot(Response_matrix[:,1,0,:],positions=np.linspace(1.5,6.5,5),widths=0.25)
edge_color='green'
for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp[element], color=edge_color)

plt.xlim((0,7))
#plt.boxplot(2*np.ones(6),Response_matrix[:,1,0,bb])
plt.ylim((-3,40))
plt.axhline(y=0)
plt.xticks(np.linspace(1.25,6.25,5),do_bands,rotation=45)

plt.ylabel('Power Change (dB)')
plt.title('Left LFP Channel')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)


ax = plt.subplot(122)
bp = plt.boxplot(Response_matrix[:,0,1,:],positions=np.linspace(1,6,5),widths=0.25)
edge_color='blue'
for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp[element], color=edge_color)

bp = plt.boxplot(Response_matrix[:,1,1,:],positions=np.linspace(1.5,6.5,5),widths=0.25)
edge_color='green'
for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
    plt.setp(bp[element], color=edge_color)

plt.xlim((0,7))

plt.ylim((-3,40))
plt.axhline(y=0)
plt.xticks(np.linspace(1.25,6.25,5),do_bands,rotation=45)


plt.ylabel('Power Change (dB)')
plt.title('Right LFP Channel')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)

plt.suptitle(band + ' Power Response to Stim')

plt.show()

#%%
#Do, in a simple way, a PCA on the entire channel space
#Focus on Alpha and Beta for now
plt.scatter(Response_matrix[:,condit,0,2],Response_matrix[:,condit,0,3])


#%%
#Do PCA on the band matrices
def Do_PCA():
    seg_subtr = 1
    
    for mm, modal in enumerate(['LFP']):
        for pp, pt in enumerate(['901','903','905','906','907','908']):
            for co, condit in enumerate(['OnTarget','OffTarget']):
                pca = []            
                pca = PCA()
                if seg_subtr:
                    start_shape = np.squeeze(SGs[modal][pt][condit]['BandSegments']['Bilat']).shape
                    flattened_matrix = np.reshape(np.squeeze(SGs[modal][pt][condit]['BandSegments']['Bilat']) - np.squeeze(SGs[modal][pt][condit]['BandSegments']['PreBilat']),(start_shape[0],start_shape[1]*start_shape[2]),'F')
                    seg = 'Bilat'
                else:
                    for sg, seg in enumerate(Ephys[modal][pt][condit]['segments'].keys()):
                        start_shape = np.squeeze(SGs[modal][pt][condit]['BandSegments'][seg]).shape
                        
                pca.fit(flattened_matrix)
                SGs[modal][pt][condit][seg]['PCA']['PCAModel'] = pca
                SGs[modal][pt][condit][seg]['PCA']['RawBands'] = flattened_matrix                
                SGs[modal][pt][condit][seg]['PCA']['RotData'] = pca.transform(flattened_matrix)
                SGs[modal][pt][condit][seg]['PCA']['VComps'] = pca.components_
                SGs[modal][pt][condit][seg]['PCA']['Evals'] = pca.explained_variance_ratio_
    
                #Plot, per patient, the timecourse of the oscillatory band powers
                #To be clear, the time dimension is removed, and each is just seen as an independent observation of the underlying process            
                #plt.figure()
                #plt.plot(flattened_matrix)
                #plt.suptitle(pt + ' ' + condit)
#%%
def Plot_PCA():
    from mpl_toolkits.mplot3d import Axes3D
    modal = 'LFP'
    pt = '901'
    condit = 'OffTarget'
    seg = 'Bilat'
    
    fig = plt.figure()
    ax = fig.add_subplot(412,projection='3d')
    ax.scatter(SGs[modal][pt][condit][seg]['PCA']['RotData'][:,0],SGs[modal][pt][condit][seg]['PCA']['RotData'][:,1],SGs[modal][pt][condit][seg]['PCA']['RotData'][:,2])
    plt.title('Top three components')
    
    ax=fig.add_subplot(411,projection='3d')
    ax.scatter(SGs[modal][pt][condit][seg]['PCA']['RawBands'][:,0],SGs[modal][pt][condit][seg]['PCA']['RawBands'][:,1],SGs[modal][pt][condit][seg]['PCA']['RawBands'][:,2])
    
    ax=fig.add_subplot(413)
    ax.imshow(SGs[modal][pt][condit][seg]['PCA']['VComps'],interpolation='none')
    
    ax=fig.add_subplot(414)
    plot_SG(modal,pt,condit,0,SGs,tpts=Ephys[modal][pt][condit]['segments'][seg])
    
    #%%
    #This gets to the meat; it starts looking at what the OffTarget data looks like in the projection of the optimal 
    big_comp_matr = np.zeros((8,8,2,6))
    eivals = np.zeros((8,2,6))
    
    for pp,pt in enumerate(['901','903','905','906','907','908']):
        fig_2 = plt.figure()
        
        ax = fig_2.add_subplot(311,projection='3d')
        temp_hold_ont = SGs[modal][pt]['OnTarget'][seg]['PCA']['RotData']
        ax.scatter(temp_hold_ont[:,0],temp_hold_ont[:,1],temp_hold_ont[:,2])
        plt.title('OnTarget Representation in OnTarget PCs')
        
        
        temp_hold = SGs[modal][pt]['OnTarget'][seg]['PCA']['PCAModel'].transform(SGs[modal][pt]['OffTarget'][seg]['PCA']['RawBands'])
        ax = fig_2.add_subplot(312,projection='3d')
        ax.scatter(temp_hold[:,0],temp_hold[:,1],temp_hold[:,2])
        plt.title('OffTarget Representation in OnTarget PCs')
        
        ax = fig_2.add_subplot(325)
        ax.imshow(SGs[modal][pt]['OnTarget'][seg]['PCA']['VComps'],interpolation='none')
        big_comp_matr[:,:,0,pp] = SGs[modal][pt]['OnTarget'][seg]['PCA']['VComps']
        eivals[:,0,pp] = SGs[modal][pt]['OnTarget'][seg]['PCA']['Evals']
        plt.title('OnTarget PCs')
    
        ax = fig_2.add_subplot(326)
        ax.imshow(SGs[modal][pt]['OffTarget'][seg]['PCA']['VComps'],interpolation='none')
        big_comp_matr[:,:,0,pp] = SGs[modal][pt]['OffTarget'][seg]['PCA']['VComps']
        eivals[:,0,pp] = SGs[modal][pt]['OffTarget'][seg]['PCA']['Evals']
        plt.title('OffTarget PCs')
        
        plt.suptitle('Patient ' + pt + ' PCA Decomp of LFP Bands')

    #%%
    plt.figure()
    plt.subplot(211)
    plt.plot(eivals[:,0,:])
    plt.legend(['DBS901','903','905','906','907','908'])
    
    plt.subplot(212)
    plt.imshow(np.mean(big_comp_matr,3)[:,:,0],interpolation='none')
    plt.title('Left Sided Vectors')
    plt.colorbar()
    axes_labels = ['L-Delta','L-Theta','L-Alpha','L-Beta*','R-Delta','R-Theta','R-Alpha','R-Beta*']
    plt.xticks(range(0,8),axes_labels,rotation='vertical')
    plt.yticks(range(0,8),axes_labels,rotation='horizontal')
    
    #%%
    #dot products for all component vectors
    #%%
    #Data saving needs to happen here
    
    #actually display the matplotlib buffer at the end
    plt.show()

#%%
#This plots the SGs for each patient individually; use this as a tool to choose segments
def plot_allpt_SGs(pt_list = ['901','903','905','906','907','908'],condit_list = ['OnTarget','OffTarget']):
    for pp,pt in enumerate(pt_list):
        #plt.figure()
        for ch in range(2):
            for cd, condit in enumerate(condit_list):
                #plt.subplot(2,2,ch + (2*(cd)+1))
                plt.figure()
                plt.subplot(211)
                plt.plot(SGs['LFP'][pt][condit]['TRaw'],SGs['LFP'][pt][condit]['Raw'])
                #plt.xlim((598,611))
                #plt.xlim((616,630))
                plt.xlim((570,670))
                
                plt.xlabel('Time (sec)')
                plt.ylabel('Amplitude (uV)')
                
                plt.subplot(212)
                #plot_SG('LFP',pt,condit,ch,SGs)
                
                plt.title(condit + ' ' + str(ch))
                #plt.xlim((598,611))
                #plt.xlim((616,630))
                #plt.xlim((570,670))
                plt.axis('off')
                
        plt.title(pt)
        
        
#%%
        
#plot_SG(disp_modal,disp_pt,disp_condit,0,SGs)
#Plot all the spectrograms here, this should be a script-specific function and should be phased out soon
#plot_allpt_SGs(['907'],['OnTarget'])
disp_modal=['LFP']
disp_pt = ['905']
disp_condit = ['OnTarget']
#for 906 bilat: timeseg = [611,690]
timeseg = [606,687]
plot_SG(disp_modal,disp_pt,disp_condit,1,SGs,tpts=timeseg)

#%%
#Extract known chirp for DBS906 and put into a file for template search in (a) voltage sweep LFP and (b) voltage sweep EEG
chirp = plot_phase(disp_modal,disp_pt,disp_condit,1,SGs,tpts=timeseg,fileio_out=False)
pickle.dump(chirp,open('/home/virati/DBS' + disp_pt[0] + '_chirp.pickle',"wb"))

#%%
chirp_templ = chirp['Raw'][1][10:30*422]
#do chirplet transform on the chirp template
pt = '905'
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



#%%

#file_writeout_the raw ts, the filtered ts, of left and right
#write_chirp(disp_modal,disp_pt,disp_condit,1,SGs,tpts=timeseg)

#%%
          
#Do the segmentation
#What times are most important?
