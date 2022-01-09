#%%
import pysindy as ps
from pysindy.feature_library import GeneralizedLibrary
import scipy.stats as stats
import json
import numpy as np
import sys
import scipy.signal as sig
import matplotlib.pyplot as plt

#%%
sys.path.append("/home/virati/Dropbox/projects/Research/MDD-DBS/Ephys/DBSpace/src/")
import DBSpace as dbo
from DBSpace import nestdict
import DBSpace.control.dyn_osc as DO

optimizer = ps.STLSQ(threshold=0.1, fit_intercept=True)
fourier_library = ps.FourierLibrary()
polynomial_library = ps.PolynomialLibrary()

functions = [lambda x: np.exp(x), lambda x, y: np.sin(x + y)]
lib_custom = ps.CustomLibrary(library_functions=functions)
lib_generalized = GeneralizedLibrary([lib_custom, fourier_library, polynomial_library])

#%%


def load_scc_lfp(pt, condit, downsample=5):
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
    #%%

    sd = np.diff(state, axis=1, append=0)

    #%%
    ## Now we get into subwindows
    do_conditions = Ephys[pt]["DOs"][0]
    pt_window = Ephys[pt][do_conditions[0]]["Configurations"][do_conditions[1]]["Stim"]

    window_idxs = np.logical_and(tvect > pt_window[0], tvect < pt_window[1])
    # Let's take out the BL stim first from the raw timeseries
    chirp = sig.decimate(state[:, window_idxs], q=downsample)

    return chirp, tvect


pt_list = ["901", "903", "905", "906", "907", "908"]
do_presence = {
    "901": ("OffTarget", "Left"),
    "903": ("OffTarget", "Left"),
    "905": ("OffTarget", "Left"),
    "906": ("OffTarget", "Right"),
    "907": ("OnTarget", "Left"),
    "908": ("OnTarget", "Left"),
}

#%%
downsample_rate = 5

for pt in ["906"]:  # pt_list:
    print(f"Analysing patient {pt}")
    DO, tvect = load_scc_lfp(pt, do_presence[pt][0], downsample_rate)
    dt = 1 / 422 * downsample_rate

    # DO[1, :] = 10 * DO[1, :]
    # DO = stats.zscore(DO,axis=1)

    plt.figure(figsize=(15, 15))
    plt.plot(DO.T)
    plt.savefig(f"ts_{pt}.png")

    coeff_gram = []
    model_scores = []
    skips = 50
    length_window = int(1 / dt * 30)

    sample_vect = np.arange(DO.shape[1])
    for tt in sample_vect[::skips]:
        if tt + length_window > DO.shape[1]:
            continue
        model = ps.SINDy(optimizer=optimizer, feature_library=lib_generalized)

        DO_snip = DO[:, tt : tt + length_window]
        model.fit(DO_snip.T, t=dt)
        coeff_gram.append(model.coefficients())
        model_scores.append(model.score(DO_snip.T))
    coeff_gram = np.array(coeff_gram)
    fig, ax = plt.subplots(figsize=(15, 15))
    # ax[0].pcolormesh()  # plot the SG here
    ax.pcolormesh(
        np.tanh(coeff_gram.reshape(coeff_gram.shape[0], -1).T / 1e3), cmap="jet"
    )
    dyn_feat_names = lib_generalized.get_feature_names()
    dyn_feat_names = ["dx0 (L):" + a for a in dyn_feat_names] + [
        "dx1 (R):" + a for a in dyn_feat_names
    ]
    ax.set_yticks(np.arange(0, 26) + 0.5)
    ax.set_yticklabels(dyn_feat_names, rotation=0)
    plt.savefig(f"sindy_coeff_{pt}.svg")
    plt.show()
