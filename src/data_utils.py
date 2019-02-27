"""
Author: Erfan Azad
Date: 22 February 2019
Description: Utility functions related to data and data generations
"""

import numpy as np


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
    for i in range(0, num_dim-1):
        for j in range(i+1,num_dim):
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
    return np.all([np.count_nonzero(p)==1 for p in x])


def all_2d(x):
    """
    Checks to see that each data point has only two non-zero
    dimension.
    Args:
        x: An -1*n ndarray.
    """
    return np.all([np.count_nonzero(p)==2 for p in x])

