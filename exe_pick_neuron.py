import os
from os.path import join as join
import h5py
import numpy as np
from scipy import io as scio
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

from utils import calculate_pearson_matrix, draw_pearson_matrix


def get_data(path):
    """ Extract data from mat files """
    data = h5py.File(path, 'r')
    key = list(data.keys())[0]
    return np.array(data.get(key)).T


def firing_curve_visualization(response_tensor, start, num, stim_index):
    """
    Visualize the firing curve of neurons, distinct the stimulus time.
    response tensor shape: [neuron, whole-time stimulus]
    """
    stim_time = 740
    stim_start = stim_index[::40, 0]
    stim_end = stim_start + stim_time
    for neuron in range(start, start + num):
        plt.subplot(num, 1, neuron+1-start)
        plt.plot(response_tensor[neuron])
        plt.vlines(stim_start, ymin=0, ymax=np.max(response_tensor[neuron]).item(), colors='g', linestyles='dashed')
        plt.vlines(stim_end, ymin=0, ymax=np.max(response_tensor[neuron]).item(), colors='r', linestyles='dashed')
        # todo: add gray background during the stimulus time, learn to use function `fill_between`
    plt.xlabel('time point')
    plt.suptitle('firing curve')
    plt.show(block=True)


def single_neuron_cls(response_tensor, cls_method, train_length, test_length):
    """
    Perform single neuron decoding to pick active neurons.
    `cls_method` belongs to sklearn package, containing knn and svm and so on.
    """
    response_tensor = response_tensor.reshape(s, t * r, -1)
    train_data = response_tensor[:, : train_length].reshape(-1, p)
    test_data = response_tensor[:, train_length:].reshape(-1, p)
    train_label = np.repeat(np.arange(4), train_length)
    test_label = np.repeat(np.arange(4), test_length)
    train_index = np.arange(train_length * s)
    np.random.shuffle(train_index)
    train_data = train_data[train_index]
    train_label = train_label[train_index]
    cls_method.fit(train_data, train_label)
    result = cls_method.predict(test_data)
    return np.sum(result == test_label) / len(tmp_test_label)


def draw_pearson(mat, t0, r0, t1, r1):
    tmp_pearson = calculate_pearson_matrix(mat[t0, r0], mat[t1, r1])
    draw_pearson_matrix(tmp_pearson, label='time bin', title='t{}r{}-t{}r{}'.format(t0, r0, t1, r1))


""" Data extraction """
basic_path = '.\\data_alter\\two_photon\\'
basic_dir_list = list(os.listdir(basic_path))
f_path = join(basic_path, 'F.mat')
f_dff_path = join(basic_path, 'F_dff.mat')
frame_between_trial_path = join(basic_path, 'frameNumBetweenTrials.mat')
start_frame_path = join(basic_path, 'startFrame.mat')
stdExt_path = join(basic_path, 'stdExt.mat')
ca_path = join(basic_path, 'CA_background.mat')

f = get_data(f_path)
f_dff = get_data(f_dff_path)
frame_between_trial = scio.loadmat(frame_between_trial_path)['frameNumBetweenTrials'].item()
# # pay attention to difference between matlab and python: the start number
start_frame = scio.loadmat(start_frame_path)['startFrame'].item()
stdExt = h5py.File(stdExt_path, 'r')
cn = np.array(stdExt.get('Cn'))
center = np.array(stdExt.get('center')).T

""" used the data of stimulus to align neural data """
ca = scio.loadmat(ca_path)
bg = ca['background'].squeeze()
thr = 1500
candidate = np.where(bg > thr)[0]
cand_dff = candidate[1:] - candidate[: -1]
cand_dff = np.concatenate([cand_dff, np.array([1])])
divider = np.where(cand_dff > 1)[0]
divider_plus = divider + 1
divider = np.sort(np.concatenate([np.array([0]), divider, divider_plus]))
index = candidate[divider]
index[1::2] += 1
final_index = index[: -1].reshape(-1, 2)

""" get the stimulus tensor """
aligned_f = []
for item in final_index[:, 0]:
    aligned_f.append(f_dff[:, item-5: item + 25])
aligned_f = np.array(aligned_f).reshape((5, 40, 822, 30))
# # shape: [type, trial, repeat, neuron, time]
f_tensor = np.concatenate([
    np.expand_dims(aligned_f[:, ::4], axis=0),
    np.expand_dims(aligned_f[:, 1::4], axis=0),
    np.expand_dims(aligned_f[:, 2::4], axis=0),
    np.expand_dims(aligned_f[:, 3::4], axis=0),
])


""" visualization for all the neurons """
start_neu = 10
firing_curve_visualization(f_dff, start_neu, 10, final_index)


""" Train classifier and pick neurons """
acc_list = []
s, t, r, n, p = f_tensor.shape
train_len = 40
test_len = 10
sel_neu_num = 100
final_neu_num = 20
tmp_test_label = np.repeat(np.arange(4), test_len)
for neuron_id in range(n):
    knn_cls = KNeighborsClassifier(n_neighbors=5)
    acc = single_neuron_cls(f_tensor[:, :, :, neuron_id], knn_cls, train_len, test_len)
    acc_list.append(acc)
acc_array = np.array(acc_list)
neu_idx = np.argsort(acc_array)[-sel_neu_num:]
f_dff_sel = f_dff[neu_idx]

# # # test
# id = np.random.randint(0, len(f_dff))
# for item in range(10):
#     knn_cls = KNeighborsClassifier(n_neighbors=5, algorithm='brute')
#     test_acc = single_neuron_cls(f_tensor[:, :, :, id], knn_cls, train_len, test_len)
#     print(test_acc)

""" visualization """
firing_curve_visualization(f_dff_sel, 90, 10, final_index)


""" post process after processing the data """
f_sel = f_tensor[:, :, :, neu_idx]
# # calculate the mean value of all the firing pattern of the neuron
f_judge = np.sum(f_sel, axis=(0, 1, 2, 4))
final_neu_index = np.argsort(f_judge)[-final_neu_num:]
raw_idx = neu_idx[final_neu_index]
f_sel_f = f_sel[:, :, :, final_neu_index]
f_dff_sel_f = f_dff[raw_idx]
firing_curve_visualization(f_dff_sel_f, 10, 10, final_index)


""" pearson matrix """
f_pro = f_sel_f
f_sel_s0 = f_pro[0]
draw_pearson(f_sel_s0, 0, 0, 0, 9)
pm_t0_r_mean = np.zeros_like(np.identity(p))
for i in range(r):
    for j in range(r):
        tmp = calculate_pearson_matrix(f_sel_s0[0, i], f_sel_s0[0, j])
        pm_t0_r_mean += tmp
pm_t0_r_mean /= r ** 2
draw_pearson_matrix(pm_t0_r_mean, label='time bin', title='rp average')

""" mean PV correlation within trials """
# calculate mean PV correlation with different repeats
mean_pv_matrix = np.identity(r)
for i in range(r):
    for j in range(r):
        tmp = calculate_pearson_matrix(f_sel_s0[0, i], f_sel_s0[0, j]).mean()
        mean_pv_matrix[i][j] = tmp
# diag = np.max(mean_pv_matrix)
# index_diag = np.identity(r).astype(np.bool_)
# mean_pv_matrix[index_diag] = diag
draw_pearson_matrix(mean_pv_matrix, label='repeat', title='mean PV correlation within trials')


""" mean PV decline within trials """
# compare the decline in the axis of elapse time
dict_mean_pv = {}
for r1 in range(r):
    for r2 in range(r1+1, r):
        tmp_pearson_mean = np.mean(calculate_pearson_matrix(f_sel_s0[0, r1], f_sel_s0[0, r2]))
        if r2 - r1 not in dict_mean_pv.keys():
            dict_mean_pv[r2 - r1] = [tmp_pearson_mean]
        else:
            dict_mean_pv[r2 - r1].append(tmp_pearson_mean)
test_pv_num_sum = 0
list_mean_pv = []
for key in dict_mean_pv.keys():
    test_pv_num_sum += len(dict_mean_pv[key])
    list_mean_pv.append(np.mean(dict_mean_pv[key]))
print(test_pv_num_sum)

for key in dict_mean_pv.keys():
    if key == r-1:
        plt.scatter([key for s in range(len(dict_mean_pv[key]))], dict_mean_pv[key], c='gray',
                    label='pvc dots')
    else:
        plt.scatter([key for s in range(len(dict_mean_pv[key]))], dict_mean_pv[key], c='gray')
plt.plot(list(range(1, len(list_mean_pv) + 1)), list_mean_pv, c='black', label='mean pv line')
plt.xticks(list(range(1, len(list_mean_pv) + 1)))
plt.xlabel('elapse time(repeat)')
plt.ylabel('mean pv correlation')
plt.title('mean pv decline within trial 0')
plt.legend()
plt.show(block=True)


""" mean PV correlation between trials """
# average across stimulus time axis
f_sel_between_trial_s0 = np.mean(f_sel_s0, axis=-1)
# shape: [5, 100, 10]
# shape: [trials, neuron, repeats]
f_sel_between_trial_s0 = np.transpose(f_sel_between_trial_s0, axes=(0, 2, 1))
matrix_over_trials = np.identity(t)
for i in range(t):
    for j in range(t):
        tmp = calculate_pearson_matrix(f_sel_between_trial_s0[i], f_sel_between_trial_s0[j]).mean()
        matrix_over_trials[i][j] = tmp
draw_pearson_matrix(matrix_over_trials, label='trials', title='mean PV correlation between trials')

""" mean PV decline between trials """
dict_mean_trial_pv = {}
for t1 in range(t):
    for t2 in range(t1+1, t):
        tmp = np.mean(calculate_pearson_matrix(f_sel_between_trial_s0[t1], f_sel_between_trial_s0[t2]))
        if t2 - t1 in dict_mean_trial_pv.keys():
            dict_mean_trial_pv[t2 - t1].append(tmp)
        else:
            dict_mean_trial_pv[t2 - t1] = [tmp]
list_mean_trial_pv = []
for key in dict_mean_trial_pv.keys():
    list_mean_trial_pv.append(np.mean(dict_mean_trial_pv[key]))
for key in dict_mean_trial_pv.keys():
    if key == t-1:
        plt.scatter([key for s in range(len(dict_mean_trial_pv[key]))], dict_mean_trial_pv[key], c='grey',
                    label='pvc dots')
    else:
        plt.scatter([key for s in range(len(dict_mean_trial_pv[key]))], dict_mean_trial_pv[key], c='grey')
plt.plot(list(range(1, len(list_mean_trial_pv)+1)), list_mean_trial_pv, c='black', label='mean pv line')
plt.xticks(list(range(1, len(list_mean_trial_pv)+1)))
plt.xlabel('elapse time(trial)')
plt.ylabel('mean pv correlation')
plt.title('mean pv decline between trials')
plt.legend()
plt.show(block=True)

""" Ensemble rate correction """
# average across time bin axis
# shape: [trials, repeat, neurons]
f_sel_ens_s0 = np.mean(f_sel_s0, axis=-1)
f_sel_ens_s0 = np.transpose(f_sel_ens_s0, axes=(0, 2, 1))
pm_sel_ens = calculate_pearson_matrix(f_sel_ens_s0[0], f_sel_ens_s0[0])
draw_pearson_matrix(pm_sel_ens, label='repeat', title='ensemble rate correlation')

""" Tuning curve correlation decline """
# shape: [10, 100, 8]
# shape: [repeat, neuron, stimulus time]
f_sel_s0_t0 = f_sel_s0[0]
dict_tuning = {}
list_tuning = []
dict_tuning_mean = {}
list_tuning_mean = []
for i in range(r):
    for j in range(i+1, r):
        list_neuron = []
        for k in range(final_neu_num):
            tmp = np.corrcoef(f_sel_s0_t0[i, k], f_sel_s0_t0[j, k])[0, 1]
            list_neuron.append(tmp)
        if j - i not in dict_tuning.keys():
            dict_tuning[j - i] = [np.nanmedian(list_neuron)]
            dict_tuning_mean[j - i] = [np.nanmean(list_neuron)]
        else:
            dict_tuning[j - i].append(np.nanmedian(list_neuron))
            dict_tuning_mean[j - i].append(np.nanmean(list_neuron))
for key in dict_tuning.keys():
    list_tuning.append(np.mean(dict_tuning[key]))
    list_tuning_mean.append(np.mean(dict_tuning_mean[key]))
plt.plot(list(range(1, len(list_tuning) + 1)), list_tuning, c='r', label='median')
plt.plot(list(range(1, len(list_tuning_mean) + 1)), list_tuning_mean, label='average')
plt.title('tuning curve correlation decline')
plt.xlabel('elapse time')
plt.ylabel('tuning curve correlation')
plt.legend()
plt.show(block=True)


""" Time-lapse decoder """
# use knn to finish classification
# shape: [stimulus time, selected_num]
stimulus_time = p
selected_num = final_neu_num
f_sel_s0_t0_trans = f_sel_s0_t0.reshape(r, stimulus_time, selected_num)
list_knn_acc = []
dict_knn_acc = {}
list_knn_shuffle = []
dict_knn_shuffle = {}
for i in range(r):
    for j in range(i+1, r):
        train = f_sel_s0_t0_trans[i]
        test = f_sel_s0_t0_trans[j]
        label = list(range(stimulus_time))
        knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
        knn.fit(train, label)
        pred = knn.predict(test)
        acc = np.sum((pred == label)) / stimulus_time
        knn_shuffle = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
        np.random.shuffle(label)
        knn_shuffle.fit(train, label)
        pred_shuffle = knn_shuffle.predict(test)
        acc_shuffle = np.sum((pred_shuffle == np.array(list(range(stimulus_time))))) / stimulus_time
        if j - i not in dict_knn_acc.keys():
            dict_knn_acc[j - i] = [acc]
            dict_knn_shuffle[j - i] = [acc_shuffle]
        else:
            dict_knn_acc[j - i].append(acc)
            dict_knn_shuffle[j - i].append(acc_shuffle)
for key in dict_knn_acc.keys():
    list_knn_acc.append(np.mean(dict_knn_acc[key]))
    list_knn_shuffle.append(np.mean(dict_knn_shuffle[key]))
plt.plot(list(range(1, len(list_knn_acc) + 1)), list_knn_acc, c='r', label='in order')
plt.plot(list(range(1, len(list_knn_shuffle) + 1)), list_knn_shuffle, c='grey', label='shuffled')
plt.xlabel('elapse time')
plt.ylabel('mean decode accuracy')
plt.title('Time-elapse decoder')
plt.legend()
plt.show(block=True)

from sklearn.decomposition import PCA
selected_r = 9
reducer_pca = PCA(n_components=3)
f_sel_pca_s0_t0_r = reducer_pca.fit_transform(f_sel_s0_t0[selected_r].reshape(stimulus_time, -1))
ax = plt.subplot(111, projection='3d')
ax.plot(f_sel_pca_s0_t0_r[:, 0], f_sel_pca_s0_t0_r[:, 1], f_sel_pca_s0_t0_r[:, 2])
ax.scatter(f_sel_pca_s0_t0_r[0, 0], f_sel_pca_s0_t0_r[0, 1], f_sel_pca_s0_t0_r[0, 2],
           c='green', label='start point')
ax.scatter(f_sel_pca_s0_t0_r[-1, 0], f_sel_pca_s0_t0_r[-1, 1], f_sel_pca_s0_t0_r[-1, 2],
           c='red', label='end point')
ax.set_title('repeat {}, pca'.format(selected_r))
plt.legend()
plt.show(block=True)

f_sel_s0_t0_r_mean = np.mean(f_sel_s0_t0, axis=0)
reducer_pca = PCA(n_components=3)
f_sel_pca_s0_t0_r_mean = reducer_pca.fit_transform(f_sel_s0_t0_r_mean.reshape(stimulus_time, -1))
ax = plt.subplot(111, projection='3d')
ax.plot(f_sel_pca_s0_t0_r_mean[:, 0], f_sel_pca_s0_t0_r_mean[:, 1], f_sel_pca_s0_t0_r_mean[:, 2])
ax.scatter(f_sel_pca_s0_t0_r_mean[0, 0], f_sel_pca_s0_t0_r_mean[0, 1], f_sel_pca_s0_t0_r_mean[0, 2],
           c='green', label='start point')
ax.scatter(f_sel_pca_s0_t0_r_mean[-1, 0], f_sel_pca_s0_t0_r_mean[-1, 1], f_sel_pca_s0_t0_r_mean[-1, 2],
           c='red', label='end point')
ax.set_title('repeat average, pca')
plt.legend()
plt.show(block=True)

