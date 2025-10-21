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
do_condits = ["OnT", "OffT"]

## Basic initialization methods, need to suppress figures from these and clean these up
eFrame = proc_dEEG.proc_dEEG(pts=pt_list, procsteps="conservative", condits=do_condits)
eFrame.standard_pipeline(blank_out_gamma=False)

# %% PSD plotting
#eFrame.plot_psd(pt="907", condit="OnT", epoch="BONT")  #'Off_3')

# %%
# Channel-marginalized Response Histogram
for pt in pt_list:
    eFrame.pop_meds(response=True, pt=pt)
    eFrame.band_distr(do_moment="mads")
    plt.suptitle(pt)

eFrame.pop_meds(response=True,pt='POOL', seg_lim=(0,10))
eFrame.band_distr(do_moment="mads")

# %%%

max_segments = np.min([len(eFrame.osc_bl_norm_timeidx[pt]['OnT']) for pt in pt_list])
#sliding_windows = [(0,5),(2,7),(4,9),(6,11),(8,13),(10,15),(12,17),(14,19),(16,21),(18,23),(20,25),(22,27),(24,29),(26,31),(28,33),(30,35),(32,37),(34,39),(36,41),(38,43),(40,45),(42,47),(44,49),(46,51),(48,53),(50,55)]
window_size = 2
sliding_windows = [(i, i + window_size) for i in range(0, max_segments - window_size + 1)]
for seg_lim in sliding_windows:
    eFrame.topo_median_response(
        do_condits=['OnT'], pt="POOL", band="Alpha", use_maya=False, seg_lim=seg_lim
    )
    plt.suptitle(f"Segment Window: {seg_lim[0]}-{seg_lim[1]}")

# Create a temporary directory to store individual frames
temp_dir = tempfile.mkdtemp()
frame_files = []

# Loop through the sliding windows and save each plot as a frame
for idx, seg_lim in enumerate(sliding_windows):
    eFrame.topo_median_response(
        do_condits=['OnT'], pt="POOL", band="Alpha", use_maya=False, seg_lim=seg_lim
    )
    plt.suptitle(f"Segment Window: {seg_lim[0]}-{seg_lim[1]}")

    # Save the current figure as a PNG
    frame_path = os.path.join(temp_dir, f"frame_{idx:03d}.png")
    plt.savefig(frame_path, dpi=100, bbox_inches='tight')
    frame_files.append(frame_path)
    plt.close()  # Close the figure to free memory

# Load all frames and create the GIF
frames = [Image.open(frame) for frame in frame_files]

# Save as GIF
output_path = "topo_median_response.gif"
frames[0].save(
    output_path,
    save_all=True,
    append_images=frames[1:],
    duration=500,  # Duration per frame in milliseconds
    loop=0  # Loop forever
)

# Clean up temporary files
for frame_file in frame_files:
    os.remove(frame_file)
os.rmdir(temp_dir)

print(f"GIF saved to: {output_path}")