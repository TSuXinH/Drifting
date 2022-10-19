import h5py
import scipy.io as scio
from os.path import join
from os import listdir

neuropixels_path = './visual_drift/data/neuropixels/'
dir_list = listdir(neuropixels_path)
neuropixels_data = scio.loadmat(join(neuropixels_path, dir_list[0]))
keys = list(neuropixels_data.keys())

cell_num = neuropixels_data['cell_num']
meaning_running_speed_repeats = neuropixels_data['mean_running_speed_repeats']
informative_rater_mat = neuropixels_data['informative_rater_mat']
neuropixels_pupils_size = neuropixels_data['valid_units_drifting_gratings']

# cell_judge_list = []
# for item in dir_list:
#     tmp = scio.loadmat(join(neuropixels_path, item))
#     cell_num = tmp['cell_num']
#     cell_judge_list.append((cell_num[0] == cell_num[1]).all())

