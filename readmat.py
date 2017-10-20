"""

readmat.py

Little script to road .mat files into Python objects.
(Specifically in this case for EEG data)

"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
import scipy.stats as stats
import util
import mne
import os

def read(matfile=""):
	""" Read in data from some .mat file """

	# OVERRIDE for testing
	harddrive = "/Volumes/Eli Hanover's WD MyPassport"
	mat1 = "/OnTarget_OffTarget_B4/less data_less noise/DBS905_B4_OffTar_HP_LP_seg_mff_cln_ref_con.mat"
	mat2 = "/OnTarget_OffTarget_B4/more data_more noise/DBS906_TO_onTAR_MU_HP_LP_seg_mff_cln_ref.mat"

	matfile = mat2

	# check filename
	assert matfile[-4:].lower() == ".mat"

	# if good, load in file
	mat = sio.loadmat(harddrive+matfile)

	#print mat
	print "keys:", mat.keys()
	for key in mat.keys():
		print type(mat[key])
		#print mat[key].shape

	return mat

#mat = read()
#print_dict(mat)




" where are these bands from and what are they? "
def topoplot(BONT_bands, BOFT_bands, sup='Title?', preplot='unity', bands=['Alpha']):
	"""
	Whaaat is going on
	What goes into topoplot for it to work?

	Not sure exactly which band I should be plotting
	"""
	plt.figure()
	plt.suptitle(sup)

	# If we want to zscore across all channels before plotting
	if preplot == 'zscore':
		preplot_f = stats.zscore
	elif preplot == 'unity':
		preplot_f = lambda x: x

	# topoplot for each band
	for i, band in enumerate(bands):
		# for each band of each time_series, find the range of values for both B
		print "Q ---> What is c????"
		set_c_max = np.max([ts[band] for ts in time_series])
		set_c_min = np.min([ts[band] for ts in time_series])

		plt.subplot(2, len(bands), i+1) # what are the last 2 parameters
		mne.viz.plot_topomap(preplot_f(BONT_bands[band]), pos=egipos.pos[:,[0,1]], vmin=set_c_min, vmax=set_c_max)
		plt.title("On Target " + band + " Changes")

		plt.subplot(2,len(bands),i+len(bands)+1)
        mne.viz.plot_topomap(preplot_f(BOFT_bands[band]),pos=egipos.pos[:,[0,1]],vmin=set_c_min,vmax=set_c_max)
        plt.title('Off Target ' + band + ' Changes')

        # ????
        # plot 3d version of BONT
		scalp_plotting(preplot_f(BONT_bands), suplabel="Mean")


BONT_bands = 
BOFT_bands = 
bont_processed = eeg.neighbors.process()
boft_processed = eeg.neighbors.process()
topoplot(BONT_bands, BOFT_bands, sup="working?")
topoplot(bont_processed, boft_processed, sup="processed?")




