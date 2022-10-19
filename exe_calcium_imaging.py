import h5py
import numpy as np
from os import listdir
import scipy.io as scio
from os.path import join
import matplotlib.pyplot as plt


base_path = './visual_drift/data/calcium_excitatory/VISp'
dir_list = listdir(base_path)
calcium_path = join(base_path, dir_list[0])  # the first

calcium_data = scio.loadmat(calcium_path)
keys = list(calcium_data.keys())

cel_reg = calcium_data['cell_registration']
cre_line = calcium_data['cre_line']
filtered_traces_days_events = calcium_data['filtered_traces_days_events']
mean_drifting_gratings_sorted = calcium_data['mean_drifting_gratings_sorted']
natural_movie_pupil_sorted = calcium_data['natural_movie_pupil_sorted']
natural_movie_running_sorted = calcium_data['natural_movie_running_sorted']
raw_pop_vector_info_trials = calcium_data['raw_pop_vector_info_trials']
united_traces_days_events = calcium_data['united_traces_days_events']
united_traces_days_spont_events = calcium_data['united_traces_days_spont_events']


# some hints from the matlab scripts
natural_movie3_traces = filtered_traces_days_events[0, 1]
# single_trace = natural_movie3_traces[0]
# x1 = single_trace[: -1].astype(np.int_)
# x2 = single_trace[1:].astype(np.int_)
# x = x2 - x1
# p1 = np.where(x == -1)[0]
# p2 = np.where(x == 1)[0]
# length = p1 - p2 + 1
# max_event_length = max(length)

trace = united_traces_days_events[0]
x = list(range(len(trace)))
plt.plot(x, trace)
plt.show()
