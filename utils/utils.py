import numpy as np


def split_data_via_stim(data, angle_num):
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


def separate_population_data(raw_data, detect_time, stimuli_time, classify_num):
    """
    Separates data according to the recorded time.
    Note that in this condition, the last stimulus will be deprecated.
    """
    data_list, label_list = [], []
    point_list = []
    for idx in range(len(stimuli_time)):
        tmp = np.where(stimuli_time[idx] - 1 < detect_time)[0][0]
        point_list.append(tmp)
    for idx in range(len(point_list) - 1):
        data_list.append(raw_data[:, point_list[idx]: point_list[idx + 1]])
        label_list.append(idx % classify_num)
    return data_list, np.array(label_list), np.array(point_list)


def align_time_dimension(data_list, point_list, min_dim=0):
    """ Aligns data with minimal or set stimulus interval. """
    interval = point_list[1:] - point_list[: -1]
    min_dim = np.min(interval) if min_dim == 0 else min_dim
    for idx in range(len(data_list)):
        data_list[idx] = data_list[idx][:, : min_dim]
    return np.array(data_list)


def separate_population_with_interval(raw_data, detect_time, stimuli_time, classify_num, interval):
    """ Separate data according to the recorded time and appointed time interval. """
    data_list, label_list = [], []
    point_list = []
    for idx in range(len(stimuli_time)):
        tmp = np.where(stimuli_time[idx] - .1 < detect_time)[0][0]
        point_list.append(tmp)
    for idx in range(len(point_list)):
        data_list.append(raw_data[:, point_list[idx]: point_list[idx] + interval])
        label_list.append(idx % classify_num)
    return np.array(data_list), np.array(label_list), np.array(point_list)


def divide_data_with_bins(data, bins):
    """ Averages the data with certain bins. """
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


def calculate_pearson_matrix(mat1, mat2):
    """ Calculates matrix of pearson correlation. """
    assert mat1.shape == mat2.shape, 'the shapes of two input matrices are not the same.'
    length = mat1.shape[-1]
    pearson_matrix = np.identity(length)
    for i in range(length):
        for j in range(length):
            pearson_matrix[i, j] = np.corrcoef(mat1[:, i], mat2[:, j])[0, 1]
    return pearson_matrix
