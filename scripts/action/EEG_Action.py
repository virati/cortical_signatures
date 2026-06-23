# %%
%load_ext autoreload
%autoreload 2
#%%
# Confirmed Working 3/29/2025

import dbspace as dbo
from dbspace.control import proc_dEEG

import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
import tempfile

import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


sns.set_context("paper")
sns.set(font_scale=2)
sns.set_style("white")
# %%
pt_list = ["906", "907", "908"]
do_condits = ["OnT", "OffT", "LeftT", "RightT"]

## Basic initialization methods, need to suppress figures from these and clean these up
eFrame = proc_dEEG.proc_dEEG(pts=pt_list, procsteps="liberal", condits=do_condits)
eFrame.standard_pipeline(blank_out_gamma=False)

# %% PSD plotting
#eFrame.plot_psd(pt="907", condit="OnT", epoch="BONT")  #'Off_3')

# %%
# Channel-marginalized Response Histogram — all stimulation conditions
for pt in pt_list:
    eFrame.pop_meds(response=True, pt=pt)
    eFrame.plot_band_distr(do_moment="mads")
    plt.suptitle(pt)

eFrame.pop_meds(response=True, pt='POOL', seg_lim=(0,10))
eFrame.band_distr(do_moment="mads")

# %%%
# Sliding-window topographic animation — one GIF per stimulation condition
stim_condits = ['OnT', 'LeftT', 'RightT']

for active_condit in stim_condits:
    max_segments = np.min([len(eFrame.osc_bl_norm_timeidx[pt][active_condit]) for pt in pt_list])
    window_size = 3
    sliding_windows = [(i, i + window_size) for i in range(0, max_segments - window_size + 1)]

    temp_dir = tempfile.mkdtemp()
    frame_files = []

    for idx, seg_lim in enumerate(sliding_windows):
        eFrame.topo_median_response(
            do_condits=[active_condit], pt="POOL", band="Alpha", use_maya=False, seg_lim=seg_lim
        )
        plt.suptitle(f"{active_condit} — Segment Window: {seg_lim[0]}-{seg_lim[1]}")

        frame_path = os.path.join(temp_dir, f"frame_{idx:03d}.png")
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        frame_files.append(frame_path)
        plt.close()

    frames = [Image.open(frame) for frame in frame_files]
    output_path = f"topo_median_response_{active_condit}.gif"
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=500,
        loop=0
    )

    for frame_file in frame_files:
        os.remove(frame_file)
    os.rmdir(temp_dir)

    print(f"GIF saved to: {output_path}")
