import mne
import readmat
import random as randy

harddrive = "/Volumes/Eli Hanover's WD MyPassport"

mat1 = "/OnTarget_OffTarget_B4/less data_less noise/DBS905_B4_OffTar_HP_LP_seg_mff_cln_ref_con.mat"
mat2 = "/OnTarget_OffTarget_B4/more data_more noise/DBS906_TO_onTAR_MU_HP_LP_seg_mff_cln_ref.mat"
mat3 = ""
mat4 = ""

bont1 = readmat.read(mat1)["BOFT"] # only get BONT bands

egipos = mne.channels.read_montage('GSN-HydroCel-257') # position of sensors

print egipos.pos[:,[0,1]]
print len(egipos.pos[:,[0,1]])


for i in range(10):
	test = [randy.randint(0, 260) for j in range(260)]
	mne.viz.plot_topomap(bont1[:,i,0], pos=egipos.pos[:,[0,1]][:-3])
