#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 23:07:30 2020

@author: virati
DO_Phase portrait and dynamics work
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
from matplotlib.gridspec import GridSpec

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
    
    #for R stim
    Ephys['906']['OffTarget']['segments']['C1'] = (368,389)
    Ephys['906']['OffTarget']['segments']['C2'] = (389,422)
    Ephys['906']['OffTarget']['segments']['C3'] = (422,475)
    Ephys['906']['OffTarget']['segments']['C4'] = (475,486)
    Ephys['906']['OffTarget']['segments']['C5'] = (488,530)

    #for bilat
    Ephys['906']['OffTarget']['segments']['C1'] = (603,615)
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
    


#%%
def scatter_phase(pt,condit):

    timeseries = dbo.load_BR_dict(Ephys[pt][condit]['Filename'],sec_offset=0)
    end_time = timeseries['Left'].shape[0]/422

    if pt == '903':
        tidxs = np.arange(231200,329300) #DBS903
    if pt == '906':
        tidxs = np.arange(256000,330200) #DBS903
    
    sos_lpf = sig.butter(10,10,output='sos',fs = 422)
    filt_L = sig.sosfilt(sos_lpf,timeseries['Left']) 
    #filt_L = sig.decimate(filt_L,2)[tidxs] #-211*60*8:
    
    filt_R = sig.sosfilt(sos_lpf,timeseries['Right'])
    #filt_R = sig.decimate(filt_R,2)[tidxs]

    #pdb.set_trace()
    plt.figure()
    plt.plot(filt_L[tidxs],filt_R[tidxs],alpha=0.1)
    t = np.linspace(0,1,filt_L[tidxs[0::50]].shape[0])
    plt.scatter(filt_L[tidxs[0::50]],filt_R[tidxs[0::50]],c=t,cmap='plasma',alpha=0.5,rasterized=True)
    plt.xlabel('Left')
    plt.ylabel('Right')
    plt.colorbar()
    
    plt.figure()
    plt.plot(filt_L[tidxs],rasterized=True)

#fundamentals('903','Left','OffTarget')
#scatter_phase('906','OffTarget')

# Now we're going to do a simple-minded 'grid' map of average change vector in each grid cell

pt = '906'
condit = 'OffTarget'

timeseries = dbo.load_BR_dict(Ephys[pt][condit]['Filename'],sec_offset=0)
end_time = timeseries['Left'].shape[0]/422

if pt == '903':
    tidxs = np.arange(231200,329300) #DBS903
if pt == '906':
    tidxs = np.arange(256000,330200) #DBS903

sos_lpf = sig.butter(10,10,output='sos',fs = 422)
filt_L = sig.sosfilt(sos_lpf,timeseries['Left']) 
#filt_L = sig.decimate(filt_L,2)[tidxs] #-211*60*8:
filt_R = sig.sosfilt(sos_lpf,timeseries['Right'])
#filt_R = sig.decimate(filt_R,2)[tidxs]

plt.figure()
plt.plot(filt_L[tidxs],filt_R[tidxs],alpha=0.1)
t = np.linspace(0,1,filt_L[tidxs[0::50]].shape[0])
plt.scatter(filt_L[tidxs[0::50]],filt_R[tidxs[0::50]],c=t,cmap='plasma',alpha=0.5,rasterized=True)
plt.xlabel('Left')
plt.ylabel('Right')
plt.colorbar()

plt.figure()
plt.plot(filt_L[tidxs],rasterized=True)

state = np.vstack((filt_L,filt_R))

sd = np.diff(state,axis=1,append=0)
min_x = np.min(state[0,:])
max_x = np.max(state[0,:])
min_y = np.min(state[1,:])
max_y = np.max(state[1,:])

xg = np.linspace(min_x,max_x,num=10)
yg = np.linspace(min_y,max_y,num=10)
#xg,yg = np.meshgrid(xg,yg)
diffgrid = np.zeros(shape=(10,10,2))
for ii in range(xg.shape[0]-1):
    for jj in range(yg.shape[0]-1):
        
        pts_in_cell = np.where(np.logical_and(np.logical_and(state[0,:] < xg[ii+1],state[0,:] > xg[ii]),np.logical_and(state[1,:] < yg[jj+1],state[1,:] > yg[jj])))
        if len(pts_in_cell[0]) != 0:
            try: changes = np.median(sd[:,pts_in_cell].squeeze(),axis=1)
            except: ipdb.set_trace()
            #pdb.set_trace()
        
            diffgrid[ii,jj,:] = changes
            #ipdb.set_trace()
        
plt.figure()
xg,yg = np.meshgrid(xg,yg)
plt.quiver(xg,yg,diffgrid[:,:,0],diffgrid[:,:,1])
#%%
import pysindy as ps

#Let's take out the BL stim first from the raw timeseries
window = np.arange(255583,296095)
chirp = state[:,window]
#plt.figure()
#plt.plot(chirp.T)

## Now we get into subwindows
subwindow_e = np.array([0,470,3500,6350,9200,12300,30000])



# if you want to plot for documents
for ii in range(subwindow_e.shape[0]-1):
    
    #sliding window linewidth
    
    chirplet = chirp[:,subwindow_e[ii]:subwindow_e[ii+1]]
    
    fig,ax = plt.subplots()
    plt.plot(chirp.T)
    plt.ylim((-0.5,1.0))
    axins = ax.inset_axes([0.5,0.5,0.5,0.5])
    axins.plot(chirp.T)
    x1,x2,y1,y2 = subwindow_e[ii],subwindow_e[ii+1],-0.4,0.4
    axins.set_xlim(x1,x2)
    axins.set_ylim(y1,y2)
    ax.indicate_inset_zoom(axins)
    #plt.plot(chirplet.T)
    
    plt.figure()
    #fig, ax = plt.subplot(2,2)
    plt.scatter(chirplet.T[:,0],chirplet.T[:,1],c=np.arange(0,chirplet.shape[1]))
    plt.plot(chirplet.T[:,0],chirplet.T[:,1])
    plt.xlim((-0.4,0.4))
    plt.ylim((-0.4,0.4))
    
    dt = 1/422
    model = ps.SINDy()
    model.fit(chirplet.T, t=dt)
    #model.print()
    
    t_test = np.arange(0, 50, dt)
    # test the prediction now
    x_sim = model.simulate(chirplet.T[0,:],t_test)
    
    #ax[1,1].subplot(2,2,2)
    plt.scatter(x_sim[:,0],x_sim[:,1])
    plt.plot(x_sim[:,0],x_sim[:,1])
    plt.text(0.1,0.1,model.print())
    
    #plt.figure()
    #plt.plot(t_test,x_sim)

#%%
if 0:
    # if you want to plot pretty HERE:
    for ii in range(subwindow_e.shape[0]-1):
        
        #sliding window linewidth
        
        chirplet = chirp[:,subwindow_e[ii]:subwindow_e[ii+1]]
        
        
        fig = plt.figure(constrained_layout=True)
        gs = GridSpec(2,2,figure=fig)
        ax1 = fig.add_subplot(gs[0,:])
        ax2 = fig.add_subplot(gs[1,0])
        ax3 = fig.add_subplot(gs[1,1])
    
        ax1.plot(chirp.T)
        #plt.plot(chirplet.T)
        
        #fig, ax = plt.subplot(2,2)
        ax2.scatter(chirplet.T[:,0],chirplet.T[:,1],c=np.arange(0,chirplet.shape[1]))
        ax2.plot(chirplet.T[:,0],chirplet.T[:,1])
        ax2.set_xlim((-0.4,0.4))
        ax2.set_ylim((-0.4,0.4))
        
        dt = 1/422
        model = ps.SINDy()
        model.fit(chirplet.T, t=dt)
        #model.print()
        
        t_test = np.arange(0, 50, dt)
        # test the prediction now
        x_sim = model.simulate(chirplet.T[0,:],t_test)
        
        #ax[1,1].subplot(2,2,2)
        ax3.scatter(x_sim[:,0],x_sim[:,1])
        ax3.plot(x_sim[:,0],x_sim[:,1])
        

        #plt.figure()
        #plt.plot(t_test,x_sim)
