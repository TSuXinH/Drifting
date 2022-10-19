import os
import h5py
import numpy as np
from copy import deepcopy
from scipy import io as scio
import matplotlib.pyplot as plt

from utils import separate_population_with_interval, \
    divide_data_with_bins, draw_pearson_matrix, calculate_pearson_matrix


def split_data_via_angle(data, angle_num):
    """
    data shape: [k * n, neuron number, time dimension]
    k is the kind of visual stimuli(number of different stimuli)
    result shape: [k, n, neuron number, time dimension]
    """
    result = []
    for angle in range(angle_num):
        tmp = list(range(angle, len(data), angle_num))
        result.append(data[tmp])
    result = np.array(result)
    return result


def pick_neuron(certain_angle_data, selected_neuron_number=0, threshold=0):
    """ pick some active neurons. """
    assert (selected_neuron_number != 0) | (threshold != 0), 'all the judgements are zero.'
    sum_data = np.sum(certain_angle_data, axis=(0, 2))  # shape: [8700]
    index = np.argsort(sum_data)
    if selected_neuron_number != 0:
        selected_index = index[-selected_neuron_number:]
        return certain_angle_data[:, selected_index], selected_index
    elif threshold != 0:
        sorted_sum_data = sum_data[index]
        selected_index_index = np.where(sorted_sum_data > threshold)[0]
        selected_index = index[selected_index_index]
        return certain_angle_data[:, selected_index], selected_index
    else:
        raise NotImplementedError


def separate_population_with_interval(raw_data, detect_time, stimuli_time, classify_num, interval, period):
    """ Separate data according to the recorded time and appointed time interval. """
    data_list, label_list = [], []
    point_list = []
    for idx in range(len(stimuli_time)):
        tmp = np.where(detect_time > stimuli_time[idx])[0][0]
        point_list.append(tmp)
    for idx in range(len(point_list)):
        data_list.append(raw_data[:, point_list[idx]: point_list[idx] + interval])
        label_list.append(idx % classify_num)
    return np.array(data_list), np.array(label_list), np.array(point_list)


def divide_data_with_bins(data, bins):
    """ Average the data with certain bins. """
    assert data.shape[-1] % bins == 0, 'Provided data can not be divided into equal bins.'
    if len(data.shape) == 2:
        length = int(data.shape[-1] // bins)
        result = data.reshape(-1, length, bins)
        result = np.mean(result, axis=-1)
        return result
    elif len(data.shape) == 3:
        batch = data.shape[1]
        length = int(data.shape[-1] // bins)
        result = data.reshape(-1, batch, length, bins)
        result = np.mean(result, axis=-1)
        return result
    else:
        return NotImplementedError


# data extraction
path_result = './3_visual_4angle/all_infered_results_filtered.mat'
path_trials = './3_visual_4angle/4_class_angle_mark_40trials.mat'
inferred_result = h5py.File(path_result, 'r')
trials = scio.loadmat(path_trials)

angle_kind = 4
stimulus_time = 40
break_interval = 10
period = .1
evt17 = trials['EVT17']  # RUSH mark, freq: 10Hz, cover all the experiment.
evt19 = trials['EVT19']  # visual stimulus angle mark, used to describe the stimulus.

keys = inferred_result.keys()
valid_c = np.array(inferred_result['valid_C']).T
valid_s = np.array(inferred_result['valid_S']).T

bins = 10  # each bin covers 1 second
# data_binned = divide_data_with_bins(data_array, bins)

# This splits the data apart, leveraging processed data.
spike_division_list = []
for item in evt19:
    tmp_idx = np.where(evt17 > item)[0][0]
    spike_division_list.append(valid_s[:, tmp_idx: tmp_idx + stimulus_time])
spike_stimulus = np.array(spike_division_list)  # shape: [160, 8700, 40]

# shape: [type, repeat,
spike_tensor = split_data_via_angle(spike_stimulus, angle_kind)  # shape: [4, 40, 8700, 40]

# # # test
# test_x = np.where(evt17 > evt19[4])[0][0]
# test_s = valid_s[:, test_x: test_x + 40]
# (test_s == spike_angle[0, 1]).all()

select_number = 50
array_sum = np.sum(spike_tensor, axis=(1, 3))
index_sorted = np.argsort(array_sum[0])
index_selected = index_sorted[-select_number:]
spike_tensor_sel = spike_tensor[:, :, index_selected]
spike_tensor_sel_s0_r0 = spike_tensor_sel[0, 0]
spike_tensor_sel_s0_r1 = spike_tensor_sel[0, 1]
spike_tensor_sel_s0_r0_bin = spike_tensor_sel_s0_r0.reshape(-1, 4, 10).mean(-1)
spike_tensor_sel_s0_r1_bin = spike_tensor_sel_s0_r1.reshape(-1, 4, 10).mean(-1)
pearson_matrix = calculate_pearson_matrix(spike_tensor_sel_s0_r0_bin, spike_tensor_sel_s0_r0_bin)
draw_pearson_matrix(pearson_matrix)

# # test
# test_p1 = valid_s[:, 1244: 1284]
# test_p2 = valid_s[:, 1444: 1484]
# test_p1_bin = test_p1.reshape(-1, 4, 10).mean(-1)
# test_p2_bin = test_p2.reshape(-1, 4, 10).mean(-1)
# pearson_matrix = calculate_pearson_matrix(test_p1_bin, test_p2_bin)
# draw_pearson_matrix(pearson_matrix)


# # population vector correlation
# # first take the absolute value, and then take all the median value by rows
# median_c = np.median(np.abs(valid_c), axis=1) * 5
# median_c_used = np.repeat(median_c.reshape(-1, 1), repeats=valid_c.shape[-1], axis=1)

# # get calcium events
# criterion1 = (valid_c - median_c_used) > 0
# criterion2 = np.concatenate(
#     [np.zeros(shape=(valid_c.shape[0], 1)), valid_c[:, 1:] - valid_c[:, : -1]], axis=1
# )
# criterion2 = criterion2 > 0
# criterion = criterion1 & criterion2
# calcium_event = np.zeros_like(valid_c)
# calcium_event[criterion] = 1.
# calcium_event[~criterion] = 0.
#
# x = 3723
#
# # calcium
# plt.subplot(311)
# single_calcium_trace = valid_c[x]
# x_tick = list(range(len(single_calcium_trace)))
# plt.plot(x_tick, single_calcium_trace)
# plt.title('calcium')
# plt.tight_layout()
#
# # spike
# plt.subplot(312)
# single_spike_trace = valid_s[x]
# plt.plot(x_tick, single_spike_trace)
# plt.title('spike')
# plt.tight_layout()
#
# # calcium events
# plt.subplot(313)
# single_calcium_event_trace = calcium_event[x]
# plt.plot(x_tick, single_calcium_event_trace)
# plt.title('calcium event')
# plt.tight_layout()
#
# plt.suptitle('neuron {}'.format(x))
# plt.tight_layout()
# plt.show(block=True)

# x = np.concatenate([np.zeros((1, )), single_trace, np.zeros((1, ))])
# x = x[1:] - x[: -1]
# p1 = np.where(x == -1)[0]
# p2 = np.where(x == 1)[0]
# length = p1 - p2 + 1
# max_length = max(length)

