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
import json

#%%
BASE_DATA_DIR = "/home/virati/Data/phd_vrt_2013/neural/imaging/DTI/MDT_DBS_2_7V_Tractography/" #NEEDS TO BE SHIFTED TO JSON
electrode_map_path = "../../assets/experiments/metadata/mayberg_900S_electrode_map.json"
DO_map_path = "../../assets/experiments/DOs/mayberg_900S_DO_map.json"

#%%
# NEED TO BUILD CONFIG LOAD HELPER FUNCTIONS HERE TODO

with open(electrode_map_path) as electrode_map_file:
    Etrode_map = json.load(electrode_map_file)
with open(DO_map_path) as DO_map_file:
    DO_map = json.load(DO_map_file)

all_pts = DO_map['parameters']['subjects']
all_condits = DO_map['parameters']['conditions']
all_sides = DO_map['parameters']['sides']
DO_all = itertools.product(all_pts,all_condits,all_sides)
DO_positive = [(item["subject"], item["condition"], item["side"]) for item in DO_map['DO_positive']]
DO_negative = [x for x in DO_all if x not in DO_positive]


#%%
sim_voltage = '2'


dti_file = nestdict()
data = nestdict()
tractos = nestdict()
combined = nestdict()

image_dim = (182,218,182)

data_arr = np.zeros((len(all_pts),len(all_condits),len(all_sides),*image_dim)) #TODO direct references instead of floating dims

for pp,pt in enumerate(all_pts):
    for cc,condit in enumerate(['OnT','OffT']):
        for ss,side in enumerate(['L','R']):
            cntct = Etrode_map[condit][pt][ss]+1
            dti_file[pp][condit][side] = BASE_DATA_DIR + 'DBS'+str(pt) + '.'+side+str(cntct)+'.' + sim_voltage + 'V.bin.nii.gz'
        
            tractos[pt][condit][side] = image.smooth_img(dti_file[pp][condit][side],fwhm=1)

            data_arr[pp,cc,ss,:,:,:] = np.array(tractos[pt][condit][side].dataobj)
                

        tractos[pt][condit]['L+R'] = image.math_img("img1+img2",img1=tractos[pt][condit]['L'],img2=tractos[pt][condit]['R'])
#%%
do_pos = nestdict()
img = [None] * len(DO_positive)
do_pos_string = ''
for aa,amalg in enumerate(DO_positive):
    img[aa] = tractos[DO_positive[aa][0]][DO_positive[aa][1]][DO_positive[aa][2]]
    do_pos_string += 'img' + str(aa) + ','
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
