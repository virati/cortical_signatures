#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 21:21:56 2018

@author: virati
This file loads in the preprocessed datafiles from AW preprocessing steps
Either the conservative versions or the non-conservative (liberal/all) versions
"""
# import sys
# sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/')
# import DBS_Osc as dbo

# from collections import defaultdict
# import mne
# from scipy.io import loadmat
# import pdb
# import numpy as np

# import scipy.stats as stats
# import matplotlib.pyplot as plt
# plt.close('all')

# from EEG_Viz import plot_3d_scalp

# import seaborn as sns
# sns.set()
# sns.set_style("white")

# from DBS_Osc import nestdict

#%%
#Simple definitions


#%%

from proc_dEEG import proc_dEEG
                
                
#%%

all_pts = ['906','907','908']

        
#UNIT TEST
SegEEG = proc_dEEG(pts=all_pts,procsteps='liberal',condits=['OnT','OffT'])
SegEEG.extract_feats()
SegEEG.compute_diff()

#%%
#Go across patients now

SegEEG.pop_response()
SegEEG.plot_pop_stats()

#%%

SegEEG.plot_diff()






#%%
#BELOW THIS IS CLUGY SHIT

if 0:
    chann_changes = np.zeros((3,2,257))
    for pp,pt in enumerate(all_pts):
        for cc,condit in enumerate(['OnT','OffT']):
            
            #SegEEG.plot_diff(pt=pt,varweigh=False,condit=condit)
            SegEEG.plot_chann_var(pt=pt,condit=condit)
            pass
        ##SegEEG.plot_ontvsofft(pt=pt)
            ##plot the max peak value
            ##Fidxs = np.where(np.logical_and(SegEEG.fvect > 0,SegEEG.fvect < 40))
            ##SegEEG.plot_topo(np.squeeze(np.max(SegEEG.PSD_diff[pt][condit][:,Fidxs],axis=2)),vmax=10,vmin=-2,label=pt + ' ' + condit)
            
            
            plot_band = 'Alpha'
            band_idx = dbo.feat_order.index(plot_band)
            #SegEEG.plot_topo(np.squeeze(SegEEG.Feat_diff[pt][condit][:,band_idx]),label=pt + ' ' + condit + ' ' + plot_band)
            threed = plt.figure()
            plot_3d_scalp(np.squeeze(SegEEG.Feat_diff[pt][condit][:,band_idx]),threed)
            
            chann_changes[pp,cc,:] = SegEEG.Feat_diff[pt][condit][:,band_idx]
    
    #%%        
    pop_channs = plt.figure()
    robust_channs = defaultdict(dict)
    for co,condit in enumerate(['OnT','OffT']):
        #plt.bar(np.arange(257),np.mean(chann_changes[pp,0,:].squeeze(),0))
        mean_change = np.mean(chann_changes[:,co,:],0)
        var_change = np.var(chann_changes[:,co,:],0)
        plt.figure(pop_channs.number)
        plt.bar(np.arange(257) + co/2,np.mean(chann_changes[:,co,:],0),yerr=np.var(chann_changes[:,co,:],0),label=condit)
        plt.xlabel('Channel Number')
        plt.ylabel('Power change in ' + plot_band)
        plt.legend()
        
        #which channels have error bars that don't cross the 0 line?
        #THIS IS KEY KNOB!
        #Right now, it keeps keeps channels where the change in power is 2dB = 1.5, AND where there is consistency across the three patients
        robust_channs[condit] = np.where(np.logical_and(np.abs(mean_change) - 7*var_change > 0,np.abs(mean_change) > 0.6)) 
        
        chann_mapper = plt.figure()
        map_channs = np.zeros((257,1))
        map_channs[robust_channs[condit]] = 1
        plt.bar(np.arange(257),map_channs.astype(float).squeeze())
        
        plt.title(condit)
        
        threed=plt.figure()
        #SegEEG.plot_topo(map_channs.reshape(-1),label='mapContacts')
        plot_3d_scalp(map_channs.reshape(-1),threed)
        plt.title(condit)
        
    #%%
    #Do above plot, but keep ALL segments available and do distributions off of that for each channel
    stack = []
    plt.figure()
    intv_key = {'OnT':'BONT','OffT':'BOFT'}
    for pt in ['906','907','908']:
        for co,condit in enumerate(['OnT','OffT']):
            stack.append(SegEEG.Feat_)
             
        
    #    pt_channvstrials = [rr[condit][intv_key[condit]][:,band_idx,:].T for (key,(key2,rr)),condit in itertools.product(SegEEG.Feat_diff.items(),[condit])]
    #    channvstrials = np.array([item for sublist in something for item in sublist])
        plt.bar(np.arange(257) + co/2,np.mean(channvstrials,0),yerr=stats.sem(channvstrials,0))
        
    #%%
    #Segment scatter plot time!
    segscatter = plt.figure()
    for cc,condit in enumerate(['OnT','OffT']):
        #all scatter data
        pass
    
    #%%
    #Quick segment time plotting
    import itertools
    plt.figure()
    [plt.plot(rr[condit]['BONT'][:,band_idx,:].T) for (key,rr),condit in itertools.product(SegEEG.Feat_trans.items(),['OnT']) if key == '906']
    
    #%%
    plt.figure()
    [plt.scatter(rr[condit]['BONT'][:,band_idx-2,:].T,rr[condit]['BONT'][:,band_idx-1,:].T,alpha=0.1) for (key,rr),condit in itertools.product(SegEEG.Feat_trans.items(),['OnT'])]
    plt.xlabel(dbo.feat_order[band_idx-2])
    plt.ylabel(dbo.feat_order[band_idx-1])
    plt.axis('equal')
    
    #%%
    # 3d scatter plotting
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    [ax.scatter(rr[condit]['BONT'][:,band_idx-2,:].T,rr[condit]['BONT'][:,band_idx-1,:].T,rr[condit]['BONT'][:,band_idx,:].T,alpha=0.1) for (key,rr),condit in itertools.product(SegEEG.Feat_trans.items(),['OnT'])]
    plt.xlabel(dbo.feat_order[band_idx-2])
    plt.ylabel(dbo.feat_order[band_idx-1])
    #plt.zlabel(dbo.feat_order[band_idx])
    
        
    #%%
    
    plt.figure()
    plt.boxplot(np.abs(chann_changes[:,0,:]))