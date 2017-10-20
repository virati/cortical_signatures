# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 10:19:45 2016

@author: virati
"""

#Oscillatory Functions and link between methods and signal processing

import scipy.signal as sig
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats.stats import pearsonr

#from sklearn import linear_model

import matplotlib
import matplotlib.pyplot as plt

import scipy.stats as stats

#from sw_mPSD_PAC import *

import pickle

from math import *

import sys
sys.path.append('/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/SigProc/CFC-Testing/Python CFC/')


font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}

matplotlib.rc('font', **font)

import pdb

patient_list = ['DBS901','DBS903','DBS905','DBS906','DBS907','DBS908']
band_dict = {'Delta':(1,4), 'Theta':(4,8), 'Alpha':(8,14), 'Beta*':(14,20), 'Beta+':(25,30), 'Gamma*':(30,50), 'Gamma+':(70,90), 'Stim':(128,132)}

def epochify(Container,sigtype='Y',epochlims=(0,-1)):
    if epochlims[1] == -1:
        epochlims[1] = Container['TS']['T'][-1]
        
    epoch_idxs = np.where(np.logical_and(Container['TS']['T'] > epochlims[0],Container['TS']['T'] < epochlims[1]))
    
    epoch = defaultdict(dict)
    epoch['Tlims'] = epochlims
    print(Container['TS'][sigtype].shape)
    epoch['Y'] = Container['TS'][sigtype][:,epoch_idxs]
    epoch['label'] = 'Generic'
    
    return epoch

def do_LPF_LFP(Container,fs=422):
    filter_lowp = sig.firwin(200,25,nyq=fs/2)
    out_sig = []
    
    for cc in range(2):
        low_filted = sig.filtfilt(filter_lowp,[1],Container['TS']['Y'][:,cc])
        out_sig.append(low_filted)
        
    return np.array(out_sig)

def plot_LFP():
    pass

def disp_LPF_LFP(Container,fs=422):
    plt.figure()
    filted_LFPs = do_LPF_LFP(Container)
    for cc, lfpsig in enumerate(filted_LFPs):
        F,T,SG = sig.spectrogram(lfpsig,nperseg=512,noverlap=256,window=sig.get_window('blackmanharris',512),fs=fs)
        plt.subplot(211)
        plt.plot(Container['TS']['T'],lfpsig)

        plt.subplot(2,2,3+cc)
        plt.pcolormesh(T,F,10*np.log10(SG))
        plt.tight_layout()
        plt.autoscale(enable=True,tight=True)
        
    plt.suptitle('Lowpass Filtered')


#This function loads in and returns a raw timeseries this needs to be phased out
def load_BR(file,Fs=422,snippet=True):
    raise ValueError('This is OBSOLETE, fix the code')
    pass

def Extract_Osc_Feats(rawdata,start_t=0,end_t=1,chann_idxs=[0,2],Fs=422):
    #start t and end_t are fractions of the total time
    channels = rawdata.shape[1]
    
    TS = defaultdict(dict)
    PSD = defaultdict(dict)
    Recording = defaultdict(dict)
    
    max_t = rawdata.shape[0]/Fs
    max_n = rawdata.shape[0]
    
    start_el = int(start_t * max_n)
    end_el = int(end_t * max_n)
    
    TS['Y'] = rawdata[start_el:end_el,chann_idxs] * 1e-3 #voltage data; this is in millivolts
        
    TS['T'] = np.linspace(start_t*max_t,end_t*max_t,TS['Y'].shape[0]) #time vector
    
    
    #Generate the PSD estimate of the TimeSeries    
    PSD['F'], PSD['LOGPxx'], PSD['P'] = comp_PSD(TS['Y'])

    #Check for stimulation on below and set the stim flag
    f_idxs = np.where(np.logical_and(PSD['F'] >= 128,PSD['F'] <= 132))
    if np.mean(np.squeeze(PSD['LOGPxx'][f_idxs,:])) > -100:
        stimflag = 1
        
    #Check what time of day the recording is from
    #time_pieces = file.split('_')
    #datetime.datetime.strptime(time_pieces[4:7],"%m/%d/%Y")
    
    
    #Return the slope for the segment
    #If we're doing snippets of data, just do the computation of the slope on the input vector
    
    #Simple oscillatory state vector added to dictionary
    OscState = defaultdict(dict)
    OVect,OBands = get_osc_state(PSD['P'],PSD['F'])
    OscState['BandVect'] = OVect
    #OscState['BandVect']['Bands'] = OBands
    #Add RMS voltage so we can
    OscState['RMS'] = np.max(np.sqrt(TS['Y']**2),axis=0)
    OscState['Pow'] = np.sum(TS['Y']**2)
    
    OscState['StimPow'] = raw_pow(PSD['P'],PSD['F'],(128,132))
    OscState['HarmPow'] = raw_pow(PSD['P'],PSD['F'],(30,34))
    OscState['TBands'] = raw_pow(PSD['P'],PSD['F'],(1,20))
    
    #return a single dictionary variable
    #Every new features will be a dictionary element, which doesn't cause issues down the line
        
    Recording['PSD'] = PSD
    Recording['TS'] = TS
    Recording['OscState'] = OscState
    #Recording['Coh'] = COH
    
    #place all flags here
    
    #THERE IS A PROBLEM HERE WHEN THERE IS ONLY ONE CHANNEL?!
    if OscState['StimPow'][0] > 0.1:
        Recording['StimFlag'] = True
    else:
        Recording['StimFlag'] = False
    #Recording['Flags'] = {'StimFlag':stimflag,'TimeFlag':timeflag}    
    
    return Recording

#This function imports the raw data, sets up a return dictionary with all the features wanted
def load_BR_feats(file,Fs=422,snippet=True,shift=0,seconds_to_take=10):
    stimflag = 0
    if snippet:
        #Snippets are always from the END
        #start_el = (-Fs * (seconds_to_take + shift))
        #We will always take to the last element
        #end_el = start_el + (Fs * 10)  - 1
        
        #for now let's just take the last half
        start_el = 0.5
        end_el = 1
        

    else:
        start_el = 0
        end_el = 1
    
    rawdata = np.array(pd.read_csv(file,sep=',',header=None))
    #Generate the TimeSeries
    #Shifts should be done here:
    return Extract_Osc_Feats(rawdata,start_el,end_el,chann_idxs=[0,2])
    
    #above should give us what we want to return; below is obsolete
    

def sliding_wind(lfp_in,nfft=2**11,fs=422,ws=5908,ss=2954):
    lfp_in=lfp_in[np.size(lfp_in/2):,[0,2]]
    slpidx=[]
    for ii in range(np.size(lfp_in)//ss):
        lfp_in=lfp_in[ii*ss:(ws+(ii*ss)),[0,2]]
        M,b=comp_slope(lfp_in)
        slpidx.append(M)
    medslp=np.median(slpidx)
    
    return medM

def comp_slope(x_in,nfft=2**11,fs=422):
    #code to compute the slope from the PSD
    
    f,P = comp_PSD(x_in,nfft=nfft,fs=fs)
    #MaybergLab Version    
    
    #What range do we want for our slopes
    f_lims = np.array([2,20])
    f_lims_idxs = np.where(np.logical_and(f >= f_lims[0], f <= f_lims[1]))
    outslope = []
    for cc in range(2):    
        M,b,rval,pval,stderr=stats.linregress(10*np.log10(f[f_lims_idxs]),P[f_lims_idxs,cc])
        outslope.append(M)
#    #Voytek Version    
#    outslope = []
#    for cc in range(2):    
#        slop = sw_mpsd_pac(x_in[:,cc],fs)
#        outslope.append(slop)
    return np.array(outslope)
    
def comp_PSD(x_in,nfft=2**11,fs=422):
    #x in is going to be a obs x chann matrix
    P = np.zeros((int(nfft/2)+1,x_in.shape[1]))
    
    for ii in range(x_in.shape[1]):
        f,P[:,ii] = sig.welch(x_in[:,ii],fs,window='blackmanharris',nperseg=512,noverlap=128,nfft=nfft)
    
    return f, 10*np.log10(np.abs(P)), P

#This is done in the form data library phase
def comp_COH(x0,x1,nfft=2**11):
    f,Cxy = sig.csd(x0,x1,fs=422,nperseg=256,window='blackmanharris',noverlap=512,nfft=nfft)
    return f,Cxy

#This is done to extract features from the already transformed data
#This needs to be a larger restructuring of the scripts and the classes
def coh_pow(X_in):
    #f,Cxy = comp_COH(X_in[],X_in[:,1])
    Coh = defaultdict(dict)    
    Coh['F'] = f
    Coh['Pxx'] = Cxy
    
    c_alpha = band_pow(Coh)
    return c_alpha

def comp_SG(x0,nfft=2**11):
    pass    

def RegModFit(exp,data,f_vect,do_pts,do_phase,side):
    import imp
    imp.reload(comp_PSD)
    #comp_PSD.LinModels(pt_PSDs,pt_HAMDs,f_vect,alpha=1)
    plt.close('ela')
    coefs = comp_PSD.Regress(exp,data,f_vect,do_pts,do_phase,alpha=0.3,side=side)
    return coefs

def Phase_OSC_Plots(pt_list,plot_phase=['B04','C20']):
    #Plot the PSD for the two timepoints
    for pp, pt in enumerate(pt_list):
        plt.figure()
        for phh, ph in enumerate(plot_phase):
            plt.plot(f_vect,pt_PSDs[:,phh,:,pp])
            plt.title('Patient ' + pt + ' PSDs')
            plt.xlim((125,135))

def plot_Phases(feat_M,feat_name,feat_list,do_pts):
    plt.figure()
    num_feats = feat_M.shape[1]
    for cc in range(2):
        for fd in range(num_feats):
            plt.subplot(num_feats,2,2*fd+cc+1)
            plt.plot(feat_M[cc,fd,:,:],alpha=0.7)
            plt.plot(np.nanmean(feat_M[cc,fd,:,:],1).T,color='black',linewidth=4)
            plt.xticks(range(0),[],rotation='vertical')
            #do similarity measure            
            plt.xlim((0,35))
            plt.title('Channel: ' + str(cc) + ' feature ' + feat_list[fd])
            plt.suptitle( feat_name + ' plot through study phases')
    
    plt.subplot(num_feats,2,2*num_feats-1)
    plt.xticks(range(0,32),Phase_List(),rotation='vertical')
    plt.subplot(num_feats,2,2*num_feats)
    plt.xticks(range(0,32),Phase_List(),rotation='vertical')
    plt.legend(do_pts)
            
def Phase_to_Matrix(ph_list,pt_list,dataF,mean_subtr=False,del_nan = False):
    EPHYS_out = np.zeros((1025,2,len(ph_list),len(pt_list)))
    HDRS_out = np.zeros((len(ph_list),len(pt_list)))
    empt_obs = []
    X = []
    Y = []
    P = []
    for phh, phase in enumerate(ph_list):
        for pp, patient in enumerate(pt_list):
            #print('Extracting from pt ' + patient)
            #print(patient + ' '  + phase + ' testing')
            
            if len(dataF[patient][phase]['MeanPSD']['LOGPxx']) > 0:
            #    EPHYS_out[:,:,phh,pp] = dataF[patient][phase]['Recordings'][0]['PSD']['Y']
                #append the recording itself                
                #X.append(dataF[patient][phase]['Recordings'][0]['PSD']['Y'])
                #append the Gold PSD
                X.append(dataF[patient][phase]['MeanPSD']['LOGPxx'])
                Y.append(dataF[patient][phase]['HDRS17'])
                #Append the HDRS scores
                #Y.append(dataF[patient][phase]['HDRS17'])
                #print(phase + ' ' + patient + '...Success')
                
            else:
                print(patient + ' ' + phase + ' does not have a recording')
                if not del_nan:
                    X.append(np.nan * np.zeros_like(dataF['DBS901']['C01']['MeanPSD']['LOGPxx']))
                    Y.append(dataF[patient][phase]['HDRS17'])
                else:
                    pass
                
            P.append(phh)
            #Flags.append()
                
            
            #    empt_obs.append((phh,pp))
            #HDRS_out[phh,pp] = dataF[patient][phase]['HDRS17']
   
    EPHYS = np.array(X)
    HDRS = np.array(Y)
    PHASES = np.array(P)
    
    #Mean0 HDRS?
    if mean_subtr:
        HDRS = HDRS - np.mean(HDRS)
        for cc in range(EPHYS.shape[0]):
            for pt in range(EPHYS.shape[2]):
                EPHYS[cc,:,pt] = EPHYS[cc,:,pt] - np.mean(EPHYS[cc,:,pt])
    return EPHYS, HDRS, PHASES

def Phase_List(exprs='all',nmo=-1):
    all_phases = ['A04','A03','A02','A01','B01','B02','B03','B04']
    for aa in range(1,25):
        if aa < 10:
            numstr = '0' + str(aa)
        else:
            numstr = str(aa)
        all_phases.append('C'+numstr)
        
        ephys_phases = all_phases[4:]
    if exprs=='all':
        return all_phases
    elif exprs=='ephys':
        return ephys_phases
    elif exprs == 'Nmo_ephys':
        #nmo = 3
        return ephys_phases[0:4*(nmo+1)-1]
    elif exprs == 'Nmo_onStim':
        #nmo = 5
        return ephys_phases[4:4*(nmo+1)-1]
        
def Flatten_Features(X):
    print(X.shape)
    #dim is (obs, features, channels)
    #We want now (obs, features x channels)
    return np.hstack((X[:,:,0],X[:,:,1]))
    


def band_pow_OBS(PSD_in,band,btype='raw'):
    band_idxs = np.where(np.logical_and(PSD_in['F'] >= band['Range'][0],PSD_in['F'] <= band['Range'][1]))
    norm_idxs = np.where(np.logical_and(PSD_in['F'] >= 4,PSD_in['F'] <= 30))
    clock_idxs = np.where(np.logical_and(PSD_in['F'] >= 105,PSD_in['F'] <= 106))
    try:
        
        base_PSD = np.squeeze(10**(PSD_in['LOGPxx']/10)) #should be 2 channels
        band_power = np.nansum(np.squeeze(base_PSD[band_idxs,:]),0)
        rel_band_power = np.nansum(np.squeeze(base_PSD[band_idxs,:]),0) / np.nansum(np.squeeze(base_PSD[norm_idxs,:]),0) #should be two numbers
        normed_band_power = np.nansum(np.squeeze(base_PSD[band_idxs,:]),0) / np.nansum(np.squeeze(base_PSD[clock_idxs,:]),0) #should be two numbers
        
        if btype == 'raw':
            return band_power
        elif btype == 'rel':
            return rel_band_power
    except:
        #print('Problem with PSD Calculation')
        return np.NaN

#This is meant to be a multidimensional/multichannel "get osc state" function
#So, coming into it, is Pxx which is a (nFFTxnChann) matrix
def get_osc_state(Pxx,F):
    band_order = ['Delta','Theta','Alpha','Beta*','Gamma*','Stim']
    #NUMBER OF CHANNELS IN Pxx
    nchann = Pxx.shape[1]
    #State is a vector
    state = np.zeros((len(band_order),nchann))
    #state is a dictionary; this is preferred
    statedict = defaultdict(dict)
    
    for bb,band in enumerate(band_order):
        bpowret = band_pow_raw(Pxx,F,band)
        assert bpowret.shape == (nchann,1)
        
        state[bb,:,None] = bpowret
        statedict[band] = state[bb]
        
    
    return statedict, band_order

#A VERY simple container that calls the real function to find the power within a particular range of frequencies
def band_pow_raw(Pxx,F,band_name,btype='raw'):
    #Get the band dictionary we are proceeding with
    #band_name is given; if the band name is not a part of the current class/library's band dictionary this will return a dict key error
    band = band_dict[band_name]
    #Actually extract the power for the given band
    #NOTE: Pxx here is STILL (nFFTxnChann)
    
    band_power = raw_pow(Pxx,F,band)
        
    return band_power

def raw_pow(Pxx,F,band):
    band_idxs = np.where(np.logical_and(F >= band[0],F <= band[1]))
    norm_idxs = np.where(np.logical_and(F >= 0,F <= 30))
    clock_idxs = np.where(np.logical_and(F >= 105,F <= 106))
    
    base_PSD = Pxx #np.squeeze(10**(Pxx/10)) #should be 2 channels
    
    #Here, we want to get a single scalar value for the power in the specified band FOR ALL nChann number of channels!
    band_power = np.sum(np.squeeze(base_PSD[band_idxs,:],axis=0),0)
    
    #If the input Pxx was actually just for one channel then we get funky behavior, like returning a (1,) array
    #which I hate, but whatever, we'll add a new axis in that instance
    
    if band_power.ndim < 2:
        band_power = band_power[:,np.newaxis]

    #So, we should get a (1 band x nChann MATRIX)    JUST SETTLE ON MATRIX
    
    #Let's assert that band-power is the shape we want it to be    
    return band_power


        
#inner product between banded power timecourse with HAMD
def pow_HDRS_sim(pow_tc,HDRS_tc):
    pow_tc = pow_tc - np.mean(pow_tc,0) #maybe 2d array
    HDRS_tc = HDRS_tc - np.mean(HDRS_tc,0) #1 d array
    
    sim_metric = np.dot(pow_tc,HDRS_tc)
    
        
def slope_pow(PSD_in):
    
    return comp_slope(PSD_in['Pxx'],nfft=2**11,fs=422)

class PSD_Collection():
    Freq_vector = []
    
    def __init__(self):
        pass
    
    def plot_PSD_HDRSlabels(self,data,pt):
        ephys_phases = Phase_List(exprs='ephys')
        X,Y,P = Phase_to_Matrix(ephys_phases,[pt],data)
        
        plt.figure()
        bcmap = matplotlib.cm.get_cmap('jet')
        freq = data['DBS901']['C01']['MeanPSD']['F']
        self.Freq_vector = freq
        
        mean_PSD = np.nanmean(np.sqrt(10**(X/10))/1e-6)
        
        print('Mean PSD Shape ' + str(mean_PSD.shape))
        for week in range(X.shape[0]):
            plt.subplot(311)
            week_plot = np.sqrt(10**(X[week,:]/10))/1e-6
            plt.semilogy(freq,week_plot,color=bcmap(week/28),linewidth=Y[week]/5)
            plt.title('PSDs per week')
            
            plt.subplot(312)
            plt.plot(freq,10*np.log10(week_plot/mean_PSD),color=bcmap(week/28))
            plt.title('Weekly PSD w/ Mean Subtraction')
            
            plt.subplot(313)
            plt.plot(freq,10*np.log10(week_plot/np.sqrt(10**(X[0,:]/10))/1e-6),color=bcmap(week/28))
            plt.title('Weekly PSD w/ First Week Subtraction')
            #plt.semilogy(freq,np.sqrt(10**(X[week,:]/10))/1e-6,color=bcmap(week/28),linewidth=Y[week]/5)
            
        #Subtract mean of all PSDs?
        Centered_matrix = 0
        plt.subplot(311)
        #Horizontal line at 150 nV/sqrt(Hz)
        plt.axhline(y=.2,linewidth=5,color='r')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (uV/rtHz)')
        plt.title('Patient: ' + pt)
        plt.xlim((0,211))
        plt.ylim((10**(-3),10**3))
        
        plt.subplot(312)
        plt.xlim((0,211))
        
        plt.subplot(313)
        plt.xlim((0,211))
        
        
        #make a return dict, with the three normalization strategies
        Chr_PSDs = {'Raw PSDs': np.sqrt(10**(X/10))/1e-6}
        return Chr_PSDs
        
    def plot_HDRSs(self,data,pt):
        pass

#a datastructure for the band dict
#band_labels = ['Delta','Theta','Alpha','Beta*','Beta+','Gamma*','Gamma+']
#band_lims = [(1,4),(4,8),(8,14),(14,20),(25,30),(30,50),(70,90)]

#A c;ass for the band dict, but this may be massive overkill
class BandDict_obsolete():
    band_labels = ['Delta','Theta','Alpha','Beta*','Beta+','Gamma*','Gamma+']
    band_lims = [(1,4),(4,8),(8,14),(14,20),(25,30),(30,50),(70,90)]
    
    def __init__(self):
        pass
    
    def returnDict(self):
        band_dict = defaultdict(dict)        
        for bb,bname in enumerate(self.band_labels):
            band_dict[bname] = self.band_lims[bb]
            
        return band_dict
        
    def Osc_feats_SG(self,Input_SG):
        return Banded_Matrix
    def Osc_feats_TS(self,Input_TS):
        pass

#%%
#For a given date and patient, return the phase it is from
def ret_phase(date,PhaseStruct):
    #List of phases to populate
    phase_list = Phase_list(exprs='ephys')
    #Need a map for each patient from their own timeline to the above phase-list
    
#%%
#Stim matrix
#This should be moved into the JSON file
def write_Stim_changes():
    stim_matr = 3.5*np.ones((6,32))
    
    stim_matr[0,16+8:] = 4.0
    stim_matr[1,10+8:] = 4.0
    stim_matr[2,8+1:] = 3.0
    stim_matr[2,8+17:] = 3.5
    stim_matr[2,8+19:] = 4.0
    stim_matr[2,8+22:] = 4.5
    stim_matr[3,8+4:] = 4.0
    stim_matr[4,8+15:] = 4.0
    stim_matr[5,8+22:] = 4.0
    
    stim_matr[:,0:8] = 0
    
    import scipy
    import scipy.io
    
    save_array = {'StimMatrix':stim_matr}
    scipy.io.savemat('/tmp/stim_changes',save_array)
    
#%%
#plot windows real quick
def plot_FFT_win(wintype='blackmanharris'):
    import scipy.signal as sig
    plt.figure()
    plt.plot(sig.get_window(wintype,512))


#%% Elastic Net Stuff
    
def ENetR(Xin,Yin,feat_side,f_trunc,feat_axis,exp,l_ratio,alpha_list,alpha=0.5,CV=False):
    from sklearn.linear_model import ElasticNet, ElasticNetCV
    from sklearn.linear_model import LinearRegression, Lasso
    X = np.array(Xin)
    Y = np.array(Yin)
    
    if feat_side == 'B':
        Input_X = np.hstack((X[:,f_trunc,0],X[:,f_trunc,1]))
    elif feat_side == 'L':
        Input_X = X[:,f_trunc,0]
    elif feat_side == 'R':
        Input_X = X[:,f_trunc,1]
    elif feat_side == 'BFull':
        Input_X = np.hstack((X[:,:,0],X[:,:,1]))
        
    EN_alpha = alpha
    n_obs = Input_X.shape
    
    if not CV:
        print('Doing vanilla Elastic Net Regression...')
        ENet = ElasticNet(alpha=EN_alpha,tol=0.001,normalize=True,positive=False)
        ENet.fit(Input_X,Y)
        ENet_error = ENet.score(Input_X,Y)

    elif CV:
        print('Doing cross-validation Elastic Net Regression...')
        k_fold = 5
        print('With k-fold value of:' + str(k_fold))    
        
        ENet = ElasticNetCV(l1_ratio=l_ratio,alphas=alpha_list,tol=0.01,normalize=True,positive=False,cv=k_fold)
        print('Input_X shape' + str(Input_X.shape))
        
        ENet.fit(Input_X,Y)
        ENet_error = ENet.score(Input_X,Y)
        print('ENet Alpha: ' + str(ENet.alpha_) + ' and L1_Ratio: ' + str(ENet.l1_ratio_))
    
    print('ENet Performance: ' + str(ENet_error))

    ENetRM = defaultdict(dict)
    ENetRM['Coefficients'] = ENet.coef_
    ENetRM['CoeffAxis'] = feat_axis
    ENetRM['Score'] = ENet_error
    ENetRM['FChannels'] = feat_side  
    ENetRM['Model'] = ENet
    ENetRM['Experiment'] = exp
    
    return ENetRM
  
def ENetPredict(ENetO,X,YActual,feat_side,f_trunc):
    #from sklearn.linear_model import ElasticNet, LinearRegression, Lasso
    if feat_side == 'B':
        Input_X = np.hstack((X[:,f_trunc,0],X[:,f_trunc,1]))
    elif feat_side == 'L':
        Input_X = X[:,f_trunc,0]
    elif feat_side == 'R':
        Input_X = X[:,f_trunc,1]    
    
    #pdb.set_trace()
    YPred = ENetO['Model'].predict(Input_X)
    #Determine score from internal function; need to better understand that score
    PredScore = ENetO['Model'].score(Input_X,YActual)
    
    #plt.figure()
    #plt.plot(ENetO['Model'].coef_)
    #plt.title('Prediction model coefficients')
    #plt.show()
    
    return YPred, PredScore
    
def ENetPlot(ENetO,doplotting=False):
    #plt.figure()
    coeff_size = ENetO['Coefficients'].shape[0]
    print('Coeff size is ' + str(coeff_size))
    DSV = defaultdict(dict)
    
    if ENetO['FChannels'] == 'B':
        DSV['Left'] = ENetO['Coefficients'][:floor(coeff_size/2)]
        DSV['Right'] = ENetO['Coefficients'][floor(coeff_size/2):]

    if doplotting:
        plt.plot(ENetO['CoeffAxis'],DSV['Left'],color='blue',linewidth=3,label='Left LFP')
        plt.plot(ENetO['CoeffAxis'],DSV['Right'],color='red',linewidth=3,label = 'Right LFP')
        
        plt.legend(loc='best')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Coefficient Magnitude')
        plt.title(ENetO['Experiment'] + ' Prediction Score: ' + str(ENetO['Score']))
        plt.suptitle('Depression Biometric (compound)')
    
    return DSV

class ENetPred():
    Model = []
    Model_Stats = 'Vanilla'
    CV = False
    StatVerif = {}
    train_ph_setup = 'ephys'
    test_ph_setup = 'Nmo_ephys' #or Nmo_ephys
    test_nph = -1
    stim_changes = []
    l_ratio = np.linspace(0.2,0.8,20)
    alpha_list = np.linspace(0.15,0.2,50)

    def __init__(self,CV=True):
        print('Initializing ENet')
        self.CV = CV
        if CV:
            self.Model_Stats = 'CV'
        
    def rem_nan(self,X,Y,P):
        isnan_idxs = np.isnan(X).any(axis=1).any(axis=1)
        print(isnan_idxs.shape)
        print(X.shape)
        X = X[~isnan_idxs]
        Y = Y[~isnan_idxs]
        P = P[~isnan_idxs]
        
        return X,Y,P
    
    def TrainEN(self,train_pts,data,f_dict,exp,alpha=0.1,stim_changes=[]):
        #Use all phases with ephys        
        train_phases = Phase_List(exprs=self.train_ph_setup)
        #Use the first three months
        #train_phases = Phase_List(exprs='3mo_ephys')
        print('Working with phases: ' + str(train_phases))
        
        X,Y,P = Phase_to_Matrix(train_phases,train_pts,data,mean_subtr=True)
        X,Y,P = self.rem_nan(X,Y,P)
        #Do combined L and R LFP regression
        feat_side = 'B'
        
        #Standard Elastic Net approach
        #If we're doing CV, we need the below two lines for parameters
        l_ratio = self.l_ratio
        alpha_list = self.alpha_list
        
        ENetModel = ENetR(X,Y,feat_side,f_dict['FreqTruncation'],f_dict['FreqVector'][f_dict['FreqTruncation']],exp = exp,alpha=alpha,CV=self.CV,l_ratio=l_ratio,alpha_list = alpha_list) #0.1 works grreat        
        DSV = ENetPlot(ENetModel,doplotting=False)
        
        #Save the elastic net model coming from above
        #pickle.dump(ENetModel,open('ElastNetModel','wb'))
        self.Model = ENetModel
        self.stim_changes = stim_changes
    
    def Stats_Scatter(self,test_pts,label_add=''):
        output_stats = defaultdict(dict)        
        print('Plotting Scatter')
        test_phases = Phase_List(exprs=self.test_ph_setup,nmo=self.test_nph)
                
        dsm_vals = stats.zscore(np.reshape(self.StatVerif['Testing']['DSM'],(len(test_pts)*len(test_phases))))
        hdrs_vals = stats.zscore(np.reshape(self.StatVerif['Testing']['HDRS'],(len(test_pts)*len(test_phases))))
        
        
        plt.subplot(211)
        pp_color = ['r','b','k','y','g','p']
        for pp,pt in enumerate(test_pts):
            plt.scatter(stats.zscore(self.StatVerif['Testing']['DSM'][pp,:]),stats.zscore(self.StatVerif['Testing']['HDRS'][pp,:]),color=pp_color[pp],label=pt)
            
        plt.xlabel('zscored DSM Value')
        plt.ylabel('zscored HDRS Value')
        
        #Do a quick linear regression, assess the relationship between the predicted depression and HDRS
        #stats.linregress
        slop,interc,rval,pval,stderr = stats.linregress(dsm_vals,hdrs_vals)
        print('Slope: ' + str(slop) + ' rval: ' + str(rval) + ' pval: ' + str(pval))
        plt.text(6, -4, 'Slope: ' + str(slop) + ' rval2: ' + str(rval**2) + ' pval: ' + str(pval), size=20, rotation=0.,
         ha="right", va="center",
         bbox=dict(boxstyle="square",
                   ec=(0, 0, 0),
                   fc=(1., 1., 1.),
                   ))
        plt.draw()
        #now plot the line
        x_regr = np.linspace(-5,5,100)        
        plt.plot(x_regr,slop*x_regr + interc,label='OLS Model',color='r',linewidth=5)
        
        plt.plot(x_regr,x_regr,label='Perfect Match',color='k')

        #Do a quick robust regression
        import statsmodels.api as sm
        from statsmodels.sandbox.regression.predstd import wls_prediction_std
        res = sm.OLS(hdrs_vals,dsm_vals).fit()
        #print(res.params)
        
        resrlm = sm.RLM(hdrs_vals,dsm_vals,M=sm.robust.norms.HuberT()).fit()
        print(resrlm.summary())
        
        plt.plot(dsm_vals,resrlm.fittedvalues,'y.-',label='RLM',linewidth=5)
        plt.legend(loc="best")
        plt.title('Scatter Plot of Prediction || ' + label_add)
                
        plt.subplot(212)
        DSV = ENetPlot(self.Model,doplotting=True)
        
        output_stats['Correspond'] = slop
        output_stats['Rval'] = rval
        output_stats['Pval'] = pval
        
        return output_stats
        
    def TestEN(self,test_pts,data,f_dict,exp,feat_side='B',nph=-1):
        print('\n\nNow doing Synthetic HDRS')
        test_phases = Phase_List(exprs=self.test_ph_setup,nmo=nph)
        self.test_nph = nph
        #self.test_phases = test_phases
        #test_pts = ['DBS905','DBS906','DBS907','DBS901','DBS903','DBS908']
        self.StatVerif['Testing'] = defaultdict(dict)
        self.StatVerif['Testing']['DSM'] = np.zeros((len(test_pts),len(test_phases)))
        self.StatVerif['Testing']['HDRS'] = np.zeros((len(test_pts),len(test_phases)))
        
        plt.figure('AllPTPred')
        for pp, tp in enumerate(test_pts):
            print('Testing in ' + tp)
            XTest,YActual,PActual = Phase_to_Matrix(test_phases,[tp],data,mean_subtr=True)
            #print(XTest.shape)
            YSynth, PredScore = ENetPredict(self.Model,XTest,YActual,feat_side,f_dict['FreqTruncation'])
            print('Prediction Score: ' + str(PredScore))
            
            from sklearn.metrics import mean_squared_error, accuracy_score
            
            plt.figure()
            plt.suptitle('Display Metrics Patient:' + tp)
            plt.subplot(311)
            plt.plot(YActual,color='blue',label="Actual HDRS-17",linewidth=5)
            #plt.xticks(range(0,32),Phase_List(exprs='ephys'),rotation='vertical')
            plt.subplot(312)
            plt.plot(YSynth,color='green',label="Predicted HDRS-17",linewidth=5)
            #plt.xticks(range(0,32),Phase_List(exprs='ephys'),rotation='vertical')
            plt.subplot(313)
            plt.figure()
            plt.plot(stats.zscore(YSynth),color='green',label="Predicted HDRS-17",linewidth=5)
            plt.plot(stats.zscore(YActual),color='blue',label="Actual HDRS-17",linewidth=5)
            if self.stim_changes != []:
                stim_change_diff = np.diff(self.stim_changes[patient_list.index(tp),4:])
                stim_change_diff[np.where(stim_change_diff > 0)] = 2
                
                #plt.plot(self.stim_changes[patient_list.index(tp),4:],color='red')
                
                plt.stem(stim_change_diff[:],'r')
                print('Printing Stim Changes for pt ' + tp + ' index ' + str(patient_list.index(tp)))
            plt.ylim((-2,2.5))
            
            plt.figure('AllPTPred')
            plt.subplot(len(test_pts),1,pp+1)
            plt.plot(stats.zscore(YActual),color='blue',label="Actual HDRS-17",linewidth=5)
            plt.plot(stats.zscore(YSynth),color='green',label="Predicted Depression State",linewidth=5)
            plt.xticks(range(0,32),Phase_List(exprs='ephys'),rotation='vertical')
            dot_simil = np.dot(stats.zscore(YActual), stats.zscore(YSynth)) / (np.linalg.norm(stats.zscore(YActual))* np.linalg.norm(stats.zscore(YSynth)))
            pear_r,pear_p = pearsonr(stats.zscore(YActual),stats.zscore(YSynth))
            plt.title('Correlation value: ' + str(pear_r) + ' w/ p-val: ' + str(pear_p))
            
            plt.legend(loc='best')
            plt.xlabel('Study Phase')
            plt.suptitle(exp + 'Patient: ' + tp + ' Prediction Score: ' + str(PredScore))
            plt.ylabel('SH Score')
            
            self.StatVerif['Testing']['DSM'][pp,:] = YSynth
            self.StatVerif['Testing']['HDRS'][pp,:] = YActual