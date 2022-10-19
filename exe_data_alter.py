import os
from os.path import join as join
import h5py
import numpy as np
from scipy import io as scio
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from utils import calculate_pearson_matrix, draw_pearson_matrix


def get_data(path):
    data = h5py.File(path, 'r')
    key = list(data.keys())[0]
    return np.array(data.get(key)).T


basic_path = '.\\data_alter\\two_photon\\'  # windows path
basic_dir_list = list(os.listdir(basic_path))
f_path = join(basic_path, 'F.mat')
f_dff_path = join(basic_path, 'F_dff.mat')
frame_between_trial_path = join(basic_path, 'frameNumBetweenTrials.mat')
start_frame_path = join(basic_path, 'startFrame.mat')
stdExt_path = join(basic_path, 'stdExt.mat')
ca_path = join(basic_path, 'CA_background.mat')

f = get_data(f_path)
f_dff = get_data(f_dff_path)
# # the data is not aligned below
f_dff_t_list = []
for i in range(1, 6):
    tmp_path = join(basic_path, 'F_dff_trial{}.mat'.format(i))
    f_dff_t_list.append(np.expand_dims(get_data(tmp_path), axis=0))
f_dff_t = np.concatenate(f_dff_t_list)

frame_between_trial = scio.loadmat(frame_between_trial_path)['frameNumBetweenTrials'].item()
# pay attention to difference between matlab and python
start_frame = scio.loadmat(start_frame_path)['startFrame'].item() - 1
stdExt = h5py.File(stdExt_path, 'r')
cn = np.array(stdExt.get('Cn'))
center = np.array(stdExt.get('center')).T
# results = np.array(stdExt.get('results'))

""" obtain background, do visualization and judge the critical point """
ca = scio.loadmat(ca_path)
bg = ca['background'].squeeze()

""" used the light data to align neural data """
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
# diff = final_index[:, 1] - final_index[:, 0]
# print(diff.max(), diff.min())
# plt.plot(bg)
# plt.scatter(index, bg[index], c='r')
# plt.show(block=True)

""" get the stimulus tensor """
aligned_f = []
for item in final_index[:, 0]:
    aligned_f.append(f_dff[:, item: item + 9])
aligned_f = np.array(aligned_f).reshape((5, 40, 822, 9))
# shape: [type, trial, repeat, neuron, time]
f_tensor = np.concatenate([
    np.expand_dims(aligned_f[:, ::4], axis=0),
    np.expand_dims(aligned_f[:, 1::4], axis=0),
    np.expand_dims(aligned_f[:, 2::4], axis=0),
    np.expand_dims(aligned_f[:, 3::4], axis=0),
])

""" The data below is not aligned """
f_divide = f_dff_t.reshape((5, 822, 10, 74))
separator = np.array([1, 9, 19, 27, 38, 46, 56, 64]).reshape(-1, 2)
f_stim = []
for idx in range(len(separator)):
    f_stim.append(f_divide[..., separator[idx, 0]: separator[idx, 1]])
f_stim = np.array(f_stim)
# # shape: [stimulus type, trial, repeat, neuron number, stimulus time]
f_tensor_alter = np.transpose(f_stim, axes=[0, 1, 3, 2, 4])  # this is the basic form of stimuli

""" Population vector correlation. """
s, t, r, n, stimulus_time = f_tensor.shape
# only see one kind of stimulus
# shape: [trial, repeat, neuron, stimulus time]
f_s0 = f_tensor_alter[0]

# pick some neurons
selected_num = 50
f_sum_wo_neuron_num = np.sum(f_s0, axis=(0, 1, 3))
index_sorted = np.argsort(f_sum_wo_neuron_num)
index_selected = index_sorted[-selected_num:]
# shape: [trial, repeat, selected neurons, stimulus time]
f_sel_s0 = f_s0[:, :, index_selected]

# # # test
# # # stimulus 2 trial 1 repeat 1
# f_test1 = f_tensor_alter[2, 3, 4]
# f_test2 = f_tensor[2, 3, 4]
# x = (f_test1 == f_test2).all()
# p1 = f_dff[:, 34: 34 + 9]
# p2 = f_dff[:, 108: 108 + 9]
# p1 = f_s0[0, 0]
# p2 = f_s0[0, 1]
# x = calculate_pearson_matrix(p1, p2)
# # draw_pearson_matrix(x)
""" PV correlation within trials """
# calculate PV correlation of two repeats
# shape: [8, 8]
# shape: [stimulus time, stimulus time]
pm_t0_r00 = calculate_pearson_matrix(f_sel_s0[1, 0], f_sel_s0[1, 0])
draw_pearson_matrix(pm_t0_r00, label='time bin', title='rp0-rp0-pm')
pm_t0_r01 = calculate_pearson_matrix(f_sel_s0[1, 0], f_sel_s0[1, 1])
draw_pearson_matrix(pm_t0_r01, label='time bin', title='rp0-rp1-pm')
pm_t0_r08 = calculate_pearson_matrix(f_sel_s0[0, 0], f_sel_s0[0, 8])
draw_pearson_matrix(pm_t0_r08, label='time bin', title='rp0-rp8-pm')
pm_t0_r09 = calculate_pearson_matrix(f_sel_s0[0, 0], f_sel_s0[0, 9])
draw_pearson_matrix(pm_t0_r09, label='time bin', title='rp0-rp9-pm')
# calculate mean PV correlation of all pair of repeats
pm_t0_r_mean = np.zeros_like(pm_t0_r00)
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
        for k in range(selected_num):
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


""" Relationship between rate and tuning stability """
#


""" similarity index """
#


""" Time-lapse decoder """
# use knn to finish classification
# shape: [stimulus time, selected_num]
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


""" decomposition """
# t-SNE
# shape: [neuron, stimulus time]
selected_r = 9
reducer = TSNE(n_components=3, init='pca', random_state=0, learning_rate=500)
f_sel_red_s0_t0_r = reducer.fit_transform(f_sel_s0_t0[selected_r].reshape(stimulus_time, -1))
ax = plt.subplot(111, projection='3d')
ax.plot(f_sel_red_s0_t0_r[:, 0], f_sel_red_s0_t0_r[:, 1], f_sel_red_s0_t0_r[:, 2])
ax.scatter(f_sel_red_s0_t0_r[0, 0], f_sel_red_s0_t0_r[0, 1], f_sel_red_s0_t0_r[0, 2],
           c='green', label='start point')
ax.scatter(f_sel_red_s0_t0_r[-1, 0], f_sel_red_s0_t0_r[-1, 1], f_sel_red_s0_t0_r[-1, 2],
           c='red', label='end point')
ax.set_title('repeat {}, t-sne'.format(selected_r))
plt.legend()
plt.show(block=True)

f_sel_s0_t0_r_mean = np.mean(f_sel_s0_t0, axis=0)
reducer = TSNE(n_components=3, init='pca', random_state=0, learning_rate=500)
f_sel_tsne_s0_t0_r_mean = reducer.fit_transform(f_sel_s0_t0_r_mean.reshape(stimulus_time, -1))
ax = plt.subplot(111, projection='3d')
ax.plot(f_sel_tsne_s0_t0_r_mean[:, 0], f_sel_tsne_s0_t0_r_mean[:, 1], f_sel_tsne_s0_t0_r_mean[:, 2])
ax.scatter(f_sel_tsne_s0_t0_r_mean[0, 0], f_sel_tsne_s0_t0_r_mean[0, 1], f_sel_tsne_s0_t0_r_mean[0, 2],
           c='green', label='start point')
ax.scatter(f_sel_tsne_s0_t0_r_mean[-1, 0], f_sel_tsne_s0_t0_r_mean[-1, 1], f_sel_tsne_s0_t0_r_mean[-1, 2],
           c='red', label='end point')
ax.set_title('repeat average, t-sne')
plt.legend()
plt.show(block=True)

# f_sel_s0_t0_tsne = f_sel_s0_t0.reshape(r * stimulus_time, selected_num)
# reducer = TSNE(n_components=2, init='pca', random_state=0, learning_rate=500, )
# f_sel_s0_t0_red = reducer.fit_transform(f_sel_s0_t0_tsne)
# f_sel_s0_t0_red = f_sel_s0_t0_red.reshape((r, stimulus_time, -1))
# for i in range(r):
#     plt.scatter(f_sel_s0_t0_red[i, :, 0], f_sel_s0_t0_red[i, :, 1], label=i+1)
# plt.legend()
# plt.show(block=True)


# PCA
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

""" internal structure """
