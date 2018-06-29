#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 21:21:56 2018

@author: virati
This file loads in the preprocessed datafiles from AW preprocessing steps
Either the conservative versions or the non-conservative (liberal/all) versions
"""
import sys
sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/')
import DBS_Osc as dbo

# from collections import defaultdict
# import mne
# from scipy.io import loadmat
# import pdb
# import numpy as np

# import scipy.stats as stats
# import matplotlib.pyplot as plt
# plt.close('all')
# import seaborn as sns
# sns.set()
# sns.set_style("white")

# from DBS_Osc import nestdict



from EEG_Viz import plot_3d_scalp
import numpy as np


from proc_dEEG import proc_dEEG
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA, FastICA

sns.set_context('paper')
sns.set(font_scale=4)
sns.set_style('white')

all_pts = ['908']#,'907','908']

        
#UNIT TEST
EEG_analysis = proc_dEEG(pts=all_pts,procsteps='conservative',condits=['OnT','OffT'])
EEG_analysis.extract_feats(polyorder=0)
EEG_analysis.gen_OSC_stack()
EEG_analysis.simple_stats()

#%%
print('Calculating Population Medians')
EEG_analysis.pop_meds()

for band in ['Alpha']:
    EEG_analysis.plot_meds(band=band,flatten=False)


#%%
EEG_analysis.band_distr('OnT')


#%%

def cSVM(EEG_analysis):
    EEG_analysis.train_SVM(mask=True)
    fig = plt.figure();plot_3d_scalp(EEG_analysis.SVM_Mask.astype(np.int),fig)
    mask_svm_coeff = EEG_analysis.SVM.coef_.reshape(3,sum(EEG_analysis.SVM_Mask),-1)
    
    
    EEG_analysis.train_SVM(mask=False)
    nomask_svm_coeff = EEG_analysis.SVM.coef_
cSVM(EEG_analysis)

#%%
#EEG_analysis.train_newGMM()
#%%
#EEG_analysis.compute_diff()

#%%
#Go across patients now

def population_stuff(SegEEG):
    print('Doing population-level stats ROUTINE')
    SegEEG.pop_response()
    
    SegEEG.do_pop_stats()
    #SegEEG.plot_pop_stats()
    
    SegEEG.plot_diff()


#population_stuff(EEG_analysis)
   
#%%
def do_similarity(SegEEG):
    print('Doing simple similarity routine')
    #generate covariance for each segments and find AVERAGE
    SegEEG.gen_GMM_dsgn(stack_bl=False)
    SegEEG.gen_GMM_feat()
    
    
    SegEEG.find_seg_covar()
    SegEEG.plot_seg_covar()
    
#do_similarity(EEG_analysis)

#%%
#SegEEG.gen_GMM_dsgn(stack_bl='normalize')
#SegEEG.gen_GMM_feat()

pca_condit = 'OnT'

def do_PCA_stuff(SegEEG):
    print('Doing NON-BASELINE PCA routine')
    #do PCA routines now
    SegEEG.pca_decomp(direction='channels',condit=pca_condit,bl_correct=False)

do_PCA_stuff(EEG_analysis)

def plot_PCA_stuff(SegEEG):
    plt.figure();
    plt.subplot(221)
    plt.imshow(SegEEG.PCA_d.components_,cmap=plt.cm.jet,vmax=1,vmin=-1)
    plt.colorbar()
    plt.subplot(222)
    plt.plot(SegEEG.PCA_d.components_)
    plt.ylim((-1,1))
    plt.legend(['PC0','PC1','PC2','PC3','PC4'])
    plt.xticks(np.arange(0,5),['Delta','Theta','Alpha','Beta','Gamma1'])
    plt.subplot(223)
    
    plt.plot(SegEEG.PCA_d.explained_variance_ratio_)
    plt.ylim((0,1))
    
    for cc in range(2):
        fig=plt.figure()
        plot_3d_scalp(SegEEG.PCA_x[:,cc],fig,animate=False,unwrap=True)
        plt.title('Plotting component ' + str(cc))
        plt.suptitle('PCA rotated results for ' + pca_condit)
plot_PCA_stuff(EEG_analysis)

#%%
#This section does MEDIANS and MADs on the data/big segment stack
#EEG_analysis.pop_meds()
print('Calculating Population Medians')
EEG_analysis.pop_meds()
#%%
for band in ['Alpha']:
    EEG_analysis.plot_meds(band=band,flatten=False)
#%%
#plot_PCA_stuff(EEG_analysis)
#EEG_analysis.plot_ICA_stuff()
EEG_analysis.do_response_PCA()
#%%
#This PCA is the baseline corrected
EEG_analysis.plot_PCA_stuff()
#%%
#Check Dynamics within segments
#THIS ASSESSED WHICH CHANNELS ARE DYNAMIC indirectly through calculating variance across OnT and OffT for all patients.

def do_DYN_assess(pEEG,band='Alpha'):
    band_idx = dbo.feat_order.index(band)
    pEEG.OnT_v_OffT_MAD()
    for stat in ['Med','MAD']:
        fig = plt.figure()
        plot_3d_scalp(pEEG.Var_Meas['OnT'][stat][:,band_idx],fig,clims=(0,0),label='OnT '+ stat,unwrap=True)
        plt.suptitle('Non-normalized Power ' + stat + ' in ' + band + ' OnT')
        
        plt.figure()
        plt.bar(np.arange(1,258),pEEG.Var_Meas['OnT'][stat][:,band_idx])
        
        fig = plt.figure()
        plot_3d_scalp(pEEG.Var_Meas['OffT'][stat][:,band_idx],fig,clims=(0,0),label='OffT ' + stat,unwrap=True)
        plt.suptitle('Non-normalized Power ' + stat + ' in ' + band + ' OffT')
        fig = plt.figure()
        plot_3d_scalp(pEEG.Var_Meas['OFF'][stat][:,band_idx],fig,clims=(0,0),label='OFF ' + stat,unwrap=True)
        plt.suptitle('Non-normalized Power ' + stat + ' in ' + band + ' OFF')
        
        
    plt.figure()
    plt.subplot(211)
    plt.hist(pEEG.Var_Meas['OnT']['Med'][:,band_idx],label='OnT',bins=30)
    plt.hist(pEEG.Var_Meas['OFF']['Med'][:,band_idx],label='OFF',bins=30)
    plt.title('Distributions of Medians')
    
    plt.subplot(212)
    plt.hist([pEEG.Var_Meas['OnT']['MAD'][:,band_idx],pEEG.Var_Meas['OFF']['MAD'][:,band_idx]],label=['OnT','OFF'],bins=30)
    #plt.hist(pEEG.Var_Meas['OFF']['MAD'][:,band_idx],label='OFF',bins=30)
    plt.title('Distributions of MADs')
    plt.legend()
    
do_DYN_assess(EEG_analysis,band='Alpha')



#%%
def do_binSVM(SegEEG):
    print('Doing Binary SVM routine')
    SegEEG.train_binSVM(mask=False)
    
    bin_coeff = SegEEG.binSVM.coef_.reshape(-1,5) #this can now go into the PCA
    for bb,band in enumerate(dbo.feat_order):
        fig = plt.figure()
        plot_3d_scalp(bin_coeff[:,bb],fig,label=band + ' SVM Coefficients',unwrap=True,animate=False)
        
    fig = plt.figure()
    plot_3d_scalp(np.linalg.norm(bin_coeff[:,:],axis=1,ord=2),fig,label=band + ' SVM Coefficients',unwrap=False,animate=False)
    plt.suptitle('L2 of all bands')
    #doing PCA on coefficients
    svm_cPCA = PCA()
    svm_cPCA.fit(bin_coeff)
    rotX = svm_cPCA.fit_transform(bin_coeff)
    
    plt.figure();
    plt.subplot(221)
    plt.imshow(svm_cPCA.components_,cmap=plt.cm.jet,vmax=1,vmin=-1)
    plt.colorbar()
    plt.subplot(222)
    plt.plot(svm_cPCA.components_)
    plt.ylim((-1,1))
    plt.legend(['PC0','PC1','PC2','PC3','PC4'])
    plt.xticks(np.arange(0,5),['Delta','Theta','Alpha','Beta','Gamma1'])
    plt.subplot(223)
    
    plt.plot(svm_cPCA.explained_variance_ratio_)
    plt.ylim((0,1))
    
    for cc in range(2):
        fig=plt.figure()
        plot_3d_scalp(rotX[:,cc],fig,animate=False,unwrap=True)
        plt.title('Plotting component ' + str(cc))
        plt.suptitle('PCA rotated results for ' + pca_condit)
    
    
do_binSVM(EEG_analysis)





#%%
GMMpreproc = False

def do_GMM_stuff(SegEEG,GMMpreproc):
    print('Doing GMM routine')
    SegEEG.train_GMM()
    return True

GMMpreproc = do_GMM_stuff(EEG_analysis,GMMpreproc)

#%%
for comp in range(2):
    plt.figure()
    plt.subplot(211)
    plt.plot(EEG_analysis.GMM.means_[comp])
    plt.subplot(212)
    if EEG_analysis.GMM.covariances_.ndim == 1:
        plt.plot(EEG_analysis.GMM.covariances_[comp,:].reshape(-1,1))
    else:
        plt.imshow(EEG_analysis.GMM.covariances_[comp,:],vmin=0,vmax=0.5)
        
        plt.colorbar()
    plt.title(comp)
    
    fig = plt.figure()
    dense_means = np.zeros((257,1))
    dense_means[EEG_analysis.median_mask==True] = EEG_analysis.GMM.means_[comp].reshape(-1,1)
    plot_3d_scalp(dense_means.squeeze(),fig,clims=(0,0.3))
    

#%%
plt.figure()
plt.plot(EEG_analysis.predictions)

l =  [[val2 for val2 in val] for key,val in EEG_analysis.GMM_stack_labels.items()]
labels = [item for sublist in l for item in sublist]
labels = [item for sublist in labels for item in sublist]

ldict = {'BONT':1,'BOFT':0}

label_numbers = [ldict[item] for item in labels]

plt.plot(label_numbers,alpha=0.7)

#compare predictions with label_numbers
correct = sum(EEG_analysis.predictions == label_numbers)
percent_correct = correct / len(label_numbers)
print('Percentage correct: ' + str(percent_correct))

plt.title('Classification of segments')
plt.xlabel('Segment number')
plt.ylabel('Class Label')
plt.yticks([0,1],['OnTarget','OffTarget'])
#%%
plt.figure()
plt.plot(EEG_analysis.posteriors)


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