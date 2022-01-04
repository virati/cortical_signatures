#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17th-ish 23:07:30 2021

@author: virati
Systematic autoepoching of DO
"""

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
from matplotlib.gridspec import GridSpec
import json

import sys

sys.path.append("/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/src/")
import DBSpace as dbo
from DBSpace import nestdict
import DBSpace.control.dyn_osc as DO


import pysindy as ps
from pysindy.feature_library import GeneralizedLibrary
import scipy.stats as stats


optimizer = ps.STLSQ(threshold=0.1, fit_intercept=True)
fourier_library = ps.FourierLibrary()
polynomial_library = ps.PolynomialLibrary()

functions = [lambda x: np.exp(x), lambda x, y: np.sin(x + y)]
lib_custom = ps.CustomLibrary(library_functions=functions)
lib_generalized = GeneralizedLibrary([lib_custom, fourier_library, polynomial_library])

#%%
def auto_epoch(pt, condit, downsample=5):
    with open(
        "../../assets/experiments/metadata/Targeting_Conditions.json", "r"
    ) as file:
        Ephys = json.load(file)

    timeseries = dbo.load_BR_dict(Ephys[pt][condit]["Filename"], sec_offset=0)
    time_lims = Ephys[pt][condit]["Configurations"]["Bilateral"]["Stim"]

    num_samples = timeseries["Left"].shape[0]
    tvect = np.linspace(0, num_samples / 422, num_samples)

    tidxs = np.logical_and(tvect > time_lims[0], tvect < time_lims[1])

    end_time = timeseries["Left"].shape[0] / 422

    sos_lpf = sig.butter(10, 20, output="sos", fs=422)
    filt_L = sig.sosfilt(sos_lpf, timeseries["Left"])
    # filt_L = sig.decimate(filt_L,2)[tidxs] #-211*60*8:
    filt_R = sig.sosfilt(sos_lpf, timeseries["Right"])
    # filt_R = sig.decimate(filt_R,2)[tidxs]

    # t = np.linspace(0, 1, filt_L[tidxs[0::50]].shape[0])

    state = np.vstack((filt_L, filt_R))

    plt.figure()
    plt.plot(tvect, state.T)
    plt.legend(["Left", "Right"])
    plt.xlim(time_lims)

    #%%

    sd = np.diff(state, axis=1, append=0)

    #%%
    ## Now we get into subwindows
    do_conditions = Ephys[pt]["DOs"][0]
    pt_window = Ephys[pt][do_conditions[0]]["Configurations"][do_conditions[1]]["Stim"]

    window_idxs = np.logical_and(tvect > pt_window[0], tvect < pt_window[1])
    # Let's take out the BL stim first from the raw timeseries
    chirp = sig.decimate(state[:, window_idxs], q=downsample)

    epoch_list = []
    current_start_idx = 0
    candidate_epochs = np.linspace(
        422 / downsample * 5, 422 / downsample * 30, 20
    ).astype(int)

    dt = 1 / 422
    done_epoching = False
    epoch_num = 0
    epoch_fit = []
    epoch_models = []
    while not done_epoching:
        epoch_num += 1
        print(f"Working on epoch {epoch_num}")
        candidate_model_score = []
        candidate_model_end_idx = []

        if current_start_idx + candidate_epochs[-1] > chirp.shape[1]:
            done_epoching = True
            continue

        for end_idx in candidate_epochs:
            epoch_try = chirp[:, current_start_idx : current_start_idx + end_idx]

            model = ps.SINDy(optimizer=optimizer, feature_library=lib_generalized)
            model.fit(epoch_try.T, t=dt)
            candidate_model_score.append(model.score(epoch_try.T))
            candidate_model_end_idx.append(end_idx)

        candidate_model_perf = np.array(candidate_model_score)
        best_model = np.argmax(candidate_model_perf)
        best_end_idx = candidate_model_end_idx[best_model]

        epoch_train = chirp[:, current_start_idx : current_start_idx + best_end_idx]

        # train the final model
        model = ps.SINDy()
        model.fit(epoch_train.T, t=dt)

        epoch_list.append((current_start_idx, current_start_idx + best_end_idx))
        current_start_idx = current_start_idx + best_end_idx
        epoch_fit.append(model.score(epoch_try.T))
        epoch_models.append(model)

    #%%
    fig, ax1 = plt.subplots()

    ax1.plot(chirp.T, alpha=0.3)
    ax1.vlines([a for (a, b) in epoch_list], -1, 1)

    fig2 = plt.figure()
    plt.plot(epoch_fit)
    plt.show()

    return epoch_list, epoch_models


def plot_epoch_models(pt, condit, epoching):
    pass


#%%
do_presence = {
    "901": ("OffTarget", "Left"),
    "903": ("OffTarget", "Left"),
    "905": ("OffTarget", "Left"),
    "906": ("OffTarget", "Right"),
    "907": ("OnTarget", "Left"),
    "908": ("OnTarget", "Left"),
}
# pt, condit = "906", "OffTarget"  # swap this out for DOs[0] from json

# pt, condit = "901", "OffTarget"

pt_list = ["901", "903", "905", "906", "907", "908"]
for pt in pt_list:
    epoching, epoch_models = auto_epoch(pt, do_presence[pt][0])
# plot_epoch_models(pt, condit, epoching)
