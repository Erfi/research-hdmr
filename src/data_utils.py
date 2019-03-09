"""
Author: Erfan Azad
Date: 22 February 2019
Description: Utility functions related to data and data generations
"""

import numpy as np
from data_classes import DatasetInfo, KeyInfo, TrainingSet


def create_1d_data(x_range, cut_center):
    """
    Args:
        x_range: A tuple of (x_start, x_end, x_step)
        cut_center: a list of num_dim

    Return:
        An ndarray of k*num_dim.
    """
    x = np.arange(*x_range)
    num_dim = len(cut_center)
    data = []
    for i in range(0, num_dim):
        for val in x:
            datum = cut_center.copy()
            datum[i] += val
            data.append(datum)
    return np.array(data)


def create_2d_data(x1_range, x2_range, cut_center):
    """
    Args:
        x1_range, x2_range: A tuple of (xi_start, xi_end, xi_step)
        cut_center: A list of size num_dim
    Return:
        An ndarray of shape k*num_dim
    """
    x1 = np.arange(*x1_range)
    x2 = np.arange(*x2_range)
    xx1, xx2 = np.meshgrid(x1, x2)
    num_dim = len(cut_center)
    data = []
    for i in range(0, num_dim - 1):
        for j in range(i + 1, num_dim):
            for k, l in zip(xx1.flatten(), xx2.flatten()):
                datum = cut_center.copy()
                datum[i] += k
                datum[j] += l
                data.append(datum)
    return np.array(data)


def all_1d(x):
    """
    Checks to see that each data point has only one non-zero
    dimension.
    Args:
        x: An -1*n ndarray.
    """
    return np.all([np.count_nonzero(p) == 1 for p in x])


def all_2d(x):
    """
    Checks to see that each data point has only two non-zero
    dimension.
    Args:
        x: An -1*n ndarray.
    """
    return np.all([np.count_nonzero(p) == 2 for p in x])


def create_data_dict(data_range, f, primary_cut_center, secondary_cut_center):
    """
    creates a dictionary of TrainingSets (n+2*nChoose2 bins)
    :param data_range: python list [start, end, step]
    :param f: ground truth function reference. f takes an ndarray of points.
    :param primary_cut_center: python list e.g. [0, 0, 1]
    :param secondary_cut_center: python list e.g. [3.5, 3, -2]
    :return: A dictionary of TrainingSet(s)
    """
    data_dict = {}
    n = len(primary_cut_center)

    # n bins for original 1D interpolation functions
    for i in range(n):
        dataset_info = DatasetInfo(data_range, primary_cut_center, i)
        key_info = KeyInfo(primary_cut_center, i)

        X = create_interpolation_data(dataset_info)
        Y = f(X).reshape(-1, 1)
        training_set = TrainingSet(X, Y)
        key = create_key(key_info)
        data_dict[key] = training_set

    # 2 * nChoose2 bins for approximate 1D interpolation functions
    for i in range(0, n - 1, 1): #
        for j in range(i + 1, n, 1):
            for k in range(2):
                if k == 0:
                    dataset_info = DatasetInfo(data_range, primary_cut_center, i, secondary_cut_center, j)
                    key_info = KeyInfo(primary_cut_center, i, secondary_cut_center, j)
                else:
                    dataset_info = DatasetInfo(data_range, primary_cut_center, j, secondary_cut_center, i)
                    key_info = KeyInfo(primary_cut_center, j, secondary_cut_center, i)
                X = create_interpolation_data(dataset_info)
                Y = f(X).reshape(-1, 1)
                training_set = TrainingSet(X, Y)
                key = create_key(key_info)
                data_dict[key] = training_set

    return data_dict


def create_interpolation_data(dataset_info: DatasetInfo) -> np.ndarray:
    """
    creates data for training one interpolation function
    :param dataset_info: An instance of DatasetInfo. Describes the data to be created.
    :return: A numpy ndarray
    """
    assert dataset_info.varying_index != dataset_info.fixed_index, 'varying index cannot be the same as the fixed index'
    single_axis_data = np.arange(*dataset_info.data_range)
    dataset = np.repeat([dataset_info.primary_cut_center], len(single_axis_data), axis=0)
    if dataset_info.secondary_cut_center:
        dataset[:, dataset_info.fixed_index] = dataset_info.secondary_cut_center[dataset_info.fixed_index]
        dataset[:, dataset_info.varying_index] = single_axis_data + dataset_info.secondary_cut_center[
            dataset_info.varying_index]
    else:
        dataset[:, dataset_info.varying_index] = single_axis_data + dataset_info.primary_cut_center[
            dataset_info.varying_index]
    return dataset


def create_key(key_info: KeyInfo) -> str:
    """
    creates a string dictionary key to access TrainingSets for each interpolation function
    :param key_info: An instance of KeyInfo
    :return: A string key
    """
    key = f'{key_info.primary_cut_center}_{key_info.varying_index}_{key_info.secondary_cut_center}_{key_info.fixed_index}'.replace(' ', '')
    return key


def create_approx_cut_center_dict(primary_cut_center, secondary_cut_center, f):
    n = len(primary_cut_center)
    cc_dict = {}
    for i in range(0, n-1, 1):
        for j in range(i+1, n, 1):
            cc = primary_cut_center.copy()
            cc[i] = secondary_cut_center[i]
            cc[j] = secondary_cut_center[j]
            key_info = KeyInfo(primary_cut_center=cc, varying_index=None)
            key = create_key(key_info)
            cc_dict[key] = f(np.array([cc]))
    return cc_dict
