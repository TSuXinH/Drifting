import itertools
import numpy as np
from matplotlib import pyplot as plt


def draw_pearson_matrix(pearson_matrix, label='', title='', saving_path=''):
    """ Show the pearson matrix. """
    plt.imshow(pearson_matrix, interpolation='nearest')
    c = 'black'
    # for i, j in itertools.product(range(pearson_matrix.shape[0]), range(pearson_matrix.shape[1])):
    #     plt.text(j, i, '{:.2f}'.format(pearson_matrix[i][j]), horizontalalignment='center', color=c)
    tick = list(range(len(pearson_matrix)))
    tick_str = list(range(1, len(pearson_matrix) + 1))
    plt.xticks(tick, tick_str)
    plt.yticks(tick, tick_str, rotation=45)
    plt.colorbar()
    plt.clim(0, 1)
    if label:
        plt.ylabel(label)
        plt.xlabel(label)
    if title:
        plt.title(title)
    if saving_path != '':
        plt.savefig(saving_path)
        plt.close()
    else:
        plt.show(block=True)
