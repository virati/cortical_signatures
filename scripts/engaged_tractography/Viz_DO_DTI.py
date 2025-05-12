# %%
%load_ext autoreload
%autoreload 2
#%%
import dbspace.control.DTI as DTI
import dbspace
from dbspace.utils.structures import nestdict
from nilearn import image, plotting
import numpy as np
import itertools


Etrode_map = dbspace.Etrode_map # THIS NEEDS TO BE SHIFTED TO JSON
BASE_DATA_DIR = "/home/virati/Data/phd_vrt_2013/neural/imaging/DTI/MDT_DBS_2_7V_Tractography/" #NEEDS TO BE SHIFTED TO JSON

#%%
all_pts = ['901','903','905','906','907','908']
all_condits = ['OnT','OffT']
all_sides = ['L','R','L+R']
voltage = '3'

DO_all = itertools.product(all_pts,all_condits,all_sides)
DO_positive = [('901','OnT','L'),
               ('901','OnT','L+R'),
               ('901','OffT','L'),
               ('901','OffT','L+R'),
               ('903','OffT','L'),
               ('903','OffT','L+R'),
               ('905','OnT','R'),
               ('905','OnT','L+R'),
               ('905','OffT','R'),
               ('905','OffT','L+R'),
               ('906','OffT','R'),
               ('906','OffT','L+R')] #This reflects the STIM conditions that evoked DOs

DO_negative = [x for x in DO_all if x not in DO_positive]

dti_file = nestdict()
data = nestdict()
tractos = nestdict()

data_arr = np.zeros((6,2,2,182,218,182))
combined = nestdict()

for pp,pt in enumerate(all_pts):
    for cc,condit in enumerate(['OnT','OffT']):
        for ss,side in enumerate(['L','R']):
            cntct = Etrode_map[condit][pt][ss]+1
            dti_file[pp][condit][side] = BASE_DATA_DIR + 'DBS'+str(pt) + '.'+side+str(cntct)+'.' + voltage + 'V.bin.nii.gz'
        
            tractos[pt][condit][side] = image.smooth_img(dti_file[pp][condit][side],fwhm=1)

            data_arr[pp,cc,ss,:,:,:] = np.array(tractos[pt][condit][side].dataobj)
                

        tractos[pt][condit]['L+R'] = image.math_img("img1+img2",img1=tractos[pt][condit]['L'],img2=tractos[pt][condit]['R'])
#%%
img = [None] * len(DO_positive)
do_pos_string = ''
for aa,amalg in enumerate(DO_positive):
    img[aa] = tractos[DO_positive[aa][0]][DO_positive[aa][1]][DO_positive[aa][2]]
    do_pos_string += 'img' + str(aa) + ','
    
do_pos = nestdict()

iter_do_pos = {'img'+str(num):img[num] for num in range(len(DO_positive))}

do_pos[condit] = image.math_img("np.mean(np.array(["+do_pos_string+"]),axis=0)",**iter_do_pos)
plotting.plot_glass_brain(do_pos[condit],black_bg=True,title='DO Positives',vmin=0,vmax=2)


#%% Now DO Negative
do_neg = nestdict()
img = [None] * len(DO_negative)
do_neg_string = ''
for aa,amalg in enumerate(DO_negative):
    img[aa] = tractos[DO_negative[aa][0]][DO_negative[aa][1]][DO_negative[aa][2]]
    do_neg_string += 'img' + str(aa) + ','

iter_do_neg= {'img'+str(num):img[num] for num in range(len(DO_negative))}

do_neg[condit] = image.math_img("np.mean(np.array(["+do_neg_string+"]),axis=0)",**iter_do_neg)
plotting.plot_glass_brain(do_neg[condit],black_bg=True,title='DO Negatives',vmin=0,vmax=2)



#%% Subtract the two somehow
diff_map = image.math_img("img0 - img1 < -0.1",img0=do_pos[condit], img1=do_neg[condit])
plotting.plot_glass_brain(diff_map,black_bg=True,title='DO Neg More',vmin=-2,vmax=2)


diff_map = image.math_img("img0 - img1 > 0.1",img0=do_pos[condit], img1=do_neg[condit])
plotting.plot_glass_brain(diff_map,black_bg=True,title='DO Pos More',vmin=-2,vmax=2)
