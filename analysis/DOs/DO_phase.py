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
import DBSpace.control.dyn_osc as DO

from matplotlib import cm
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.signal as sig
from matplotlib.gridspec import GridSpec

#%%
Ephys = DO.Ephys
Phase = 'TurnOn'

pt = '906'
condit = 'OffTarget'

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

#%%
plt.figure()
plt.subplot(211)
plt.plot(timeseries['Left'])
plt.plot(timeseries['Right'])

plt.subplot(212)
plt.plot(filt_L[tidxs],rasterized=True)
plt.plot(filt_R[tidxs],rasterized=True)


state = np.vstack((filt_L,filt_R))
sd = np.diff(state,axis=1,append=0)

#%%
if 0:
    # A vector field approach 
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
import scipy.stats as stats

plt.close('all')
## Now we get into subwindows
pt_windows = {'906':np.arange(255550,296095), '903':np.arange(231200,329300)}
pt_regimes= {'906':np.array([0,800,3500,6350,9200,12300,30000]),'903':np.array([0,1470,6260,27020,80000,97940])}

window = pt_windows[pt]
subwindow_e = pt_regimes[pt]

#Let's take out the BL stim first from the raw timeseries
chirp = sig.decimate(state[:,window],q=1)
#plt.figure()
#plt.plot(chirp.T)

#chirp[0,:] = stats.zscore(chirp[0,:])
#chirp[1,:] = stats.zscore(chirp[1,:])

coeffs = []
# if you want to plot for documents
for ii in range(subwindow_e.shape[0]-1):
    print(ii)
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
    plt.plot(chirplet.T[:,0],chirplet.T[:,1],alpha=0.2)
    #plt.xlim((-0.4,0.4))
    #plt.ylim((-0.4,0.4))
    
    dt = 1/422
    model = ps.SINDy()
    model.fit(chirplet.T, t=dt)
    #model.print()
    
    t_test = np.arange(0, 50, dt)
    # test the prediction now
    x_sim = model.simulate(chirplet.T[0,:],t_test)
    
    #ax[1,1].subplot(2,2,2)
    #plt.scatter(x_sim[:,0],x_sim[:,1])
    plt.plot(x_sim[:,0],x_sim[:,1],linewidth=2,alpha=1)
    plt.text(0.1,0.1,model.print())
    
    
    coeffs.append(model.coefficients().reshape(-1,1))
    #%%
plt.figure()
plt.imshow(np.array(coeffs).squeeze().T,clim=(-5,5))
plt.colorbar(cmap='jet')
plt.title(pt + ' '  + condit)
    
    #plt.figure()
    #plt.plot(t_test,x_sim)