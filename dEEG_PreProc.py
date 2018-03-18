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

from EEG_Viz import plot_3d_scalp

# import seaborn as sns
# sns.set()
# sns.set_style("white")

# from DBS_Osc import nestdict

#%%
#Simple definitions


#%%

from proc_dEEG import proc_dEEG
import matplotlib.pyplot as plt
   
all_pts = ['906','907','908']

        
#UNIT TEST
EEG_analysis = proc_dEEG(pts=all_pts,procsteps='conservative',condits=['OnT','OffT'])
EEG_analysis.extract_feats()
EEG_analysis.compute_diff()

#%%
#Go across patients now

def population_stuff(SegEEG):
    
    SegEEG.pop_response()
    
    SegEEG.do_pop_stats()
    #SegEEG.plot_pop_stats()
    
    SegEEG.plot_diff()


population_stuff(EEG_analysis)
   
#%%
def do_similarity(SegEEG):
    #generate covariance for each segments and find AVERAGE
    SegEEG.gen_GMM_dsgn(stack_bl=False)
    SegEEG.gen_GMM_feat()
    
    
    SegEEG.find_seg_covar()
    SegEEG.plot_seg_covar()
    
do_similarity(EEG_analysis)

#%%
#SegEEG.gen_GMM_dsgn(stack_bl='normalize')
#SegEEG.gen_GMM_feat()
    
def do_PCA_stuff(SegEEG):
    
    #do PCA routines now
    SegEEG.pca_decomp()
    
    plt.figure();
    plt.subplot(221)
    plt.imshow(EEG_analysis.PCA_d.components_)
    plt.subplot(222)
    plt.plot(EEG_analysis.PCA_d.components_)
    plt.legend({'PC1','PC2','PC3','PC4'})
    plt.xticks(np.arange(0,5),['Delta','Theta','Alpha','Beta','Gamma'])
    plt.subplot(223)
    plt.imshow(EEG_analysis.PCA_x)
    
    plt.figure()
    plt.plot(EEG_analysis.PCA_d.explained_variance_ratio_)
    
    for cc in range(5):
        fig=plt.figure()
        plot_3d_scalp(EEG_analysis.PCA_x[:,cc],fig)
        plt.title('Plotting component ' + str(cc))
        plt.suptitle('PCA rotated results for OnTarget')

do_PCA_stuff(EEG_analysis)
#%%
GMMpreproc = False

#%%
def do_GMM_stuff(SegEEG,GMMpreproc):
    if not GMMpreproc:
        SegEEG.gen_GMM_dsgn(stack_bl='normalize')
        SegEEG.gen_GMM_feat()
        SegEEG.pop_meds()
    
    SegEEG.train_GMM()
    return True

GMMpreproc = do_GMM_stuff(EEG_analysis,GMMpreproc)

plt.figure()
plt.plot(EEG_analysis.Seg_Med[0]['OnT'],label='OnT')
plt.plot(EEG_analysis.Seg_Med[0]['OffT'],label='OffT')
plt.title('Medians across Channels')
plt.legend()
for condit in EEG_analysis.condits:
    fig = plt.figure()
    plot_3d_scalp(EEG_analysis.Seg_Med[0][condit],fig,label=condit + '_med',animate=False)
    plt.suptitle('Median of all channels across all ' + condit + ' segments')

plt.figure()
plt.plot(EEG_analysis.Seg_Med[1]['OnT'],label='OnT')
plt.plot(EEG_analysis.Seg_Med[1]['OffT'],label='OffT')
plt.title('MADs across Channels')
plt.legend()
for condit in EEG_analysis.condits:
    fig = plt.figure()
    plot_3d_scalp(EEG_analysis.Seg_Med[1][condit],fig,label=condit + '_mad',animate=False)
    plt.suptitle('MADs of all channels across all ' + condit + ' segments')
#%%

for comp in range(2):
    plt.figure()
    plt.subplot(211)
    plt.plot(EEG_analysis.GMM.means_[comp])
    plt.subplot(212)
    plt.imshow(EEG_analysis.GMM.covariances_[comp,:].reshape(-1,1),vmin=0,vmax=0.2)
    plt.colorbar()
    plt.title(comp)
    
    fig = plt.figure()
    plot_3d_scalp(EEG_analysis.GMM.means_[comp],fig,clims=(0,4))
    

#%%
plt.figure()
plt.plot(EEG_analysis.predictions)

l =  [[val2 for val2 in val] for key,val in EEG_analysis.GMM_stack_labels.items()]
labels = [item for sublist in l for item in sublist]
labels = [item for sublist in labels for item in sublist]

ldict = {'BONT':0,'BOFT':1}

label_numbers = [ldict[item] for item in labels]

plt.plot(label_numbers,alpha=0.7)
plt.title('Classification of segments')
plt.xlabel('Segment number')
plt.ylabel('Class Label')
plt.yticks([0,1],['OnTarget','OffTarget'])


#%%
def population_3d_topos(SegEEG):
    do_bands = ['Delta','Theta','Alpha','Beta','Gamma1']
    do_bands = ['Alpha']
    for condit in SegEEG.condits:
        for band in do_bands:
            #Mask sets a threshold based off of VARIANCE across the three patients
            SegEEG.topo_wrap(band=band,condit=condit,label=condit + ' ' + band,mask=False,animate=False)
    

#population_3d_topos(EEG_analysis)


#%%
#Do simple classifier
def simple_classif(SegEEG):
    SegEEG.train_simple()
    OnT,OffT = SegEEG.test_simple()
    
    return OnT,OffT
    
OnT, OffT = simple_classif(EEG_analysis)

#%% 
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