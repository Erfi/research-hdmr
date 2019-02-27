"""
Author: Erfan Azad
Date: 22 February 2019
Description: The core algorithms and related functions
"""
import numpy as np
from scipy.interpolate import interp1d, interp2d

from data_utils import all_1d, all_2d


def create_interp_1d_funcs(x, y, cut_center):
    """
    Args:
        x: An k*n ndarray. For k examples of n-dimentional data.
            Training data points must be one-dimentional
            e.g. [0.9, 0, 0], [-2, 0, 0] for i = 0
        y: A k*1 ndarray of label (regression) values for each k points
        cut_center: An 1*n ndarray for the cut_center point

    Returns:
        List of n 1d interpolation functions.
        Index i in the list represents F_Interp_i(xi_new)
    """
    n = x.shape[1]  # number of dimentions
    assert n > 1, 'interp 1d functions require at least 2 dimentional datapoints'
    x_adjusted = x - cut_center  # adjusting data points s.t. cut_center becomes the origin
    interp_funcs = []  # holding all n interpolation functions
    for i in range(n):
        indx = np.where(x_adjusted[:, i] != 0)  # index of adjusted datapoints with value at dim i

        x_adjusted_selected = x_adjusted[indx]  # actual datapoints with value at dim i
        assert all_1d(x_adjusted_selected), 'Training Data is not 1D'  # making sure that only dim i has data

        x_selected = x[indx]  # original datapoints for dim i data
        y_selected = y[indx]  # original labels for dim i data

        x_col = x_selected[:, i]  # ith dim of selected datapoints (only the values not 0s)
        fi = interp1d(x_col.flatten(), y_selected.flatten(), kind='cubic')  # interpolating function for dim i

        interp_funcs.append(fi)
    return interp_funcs


def create_interp_2d_funcs(x, y, cut_center):
    """
    Args:
        x: An k*n ndarray. For k examples of n-dimentional data.
            Training data points must be one-dimentional
            e.g. [0.9, 0, 0], [-2, 0, 0] for i = 0
        y: A k*1 ndarray of label (regression) values for each k points
        cut_center: An 1*n ndarray for the cut_center point

    Returns:
        List of n 1d interpolation functions.
        Index i in the list represents F_Interp_i(xi_new)
    """
    n = x.shape[1]
    assert n > 2  # interp 2d functions require at least 3 dimentional data
    x_adjusted = x - cut_center
    interp_functions = []
    for i in range(0, n - 1):
        func_row = []
        for j in range(i + 1, n):
            indx = np.where((x_adjusted[:, i] != 0) & (x_adjusted[:, j] != 0))

            x_adjusted_selected = x_adjusted[indx]
            assert all_2d(x_adjusted_selected), 'Training data is not 2d'

            x_selected = x[indx]
            y_selected = y[indx].tolist()

            x_col_1 = x_selected[:, i].tolist()
            x_col_2 = x_selected[:, j].tolist()
            fi = interp2d(x_col_1, x_col_2, y_selected, kind='cubic')
            func_row.append(fi)
        interp_functions.append(func_row)
    return interp_functions


def sigma_f1d(f0, f1d, datum):
    """
    Args:
        f0: function value at the cut center
        f1d: list of 1d interpolation functions
        datum: test data. Pyhton list.
    """
    num_funcs = len(f1d)
    total = 0
    for i in range(num_funcs):
        total += f1d[i](datum[i]) - f0
    return total


def sigma_f2d(f0, f1d, f2d, datum):
    """
    Args:
        f0: function value at the cut center
        f1d: list of n 1d functions (n: total dimention of the data points)
        f2d: list of list of 2d interpolation funcitons
            s.t. [[f12, f13, f14],[f23, f24], [f34]] for n=4
        datum: test datum. a python list.
    """
    total = 0
    for i in range(len(f2d)):
        for j in range(len(f2d[i])):
            dim1_index = i
            dim2_index = i + 1 + j
            total += f2d[i][j](datum[dim1_index], datum[dim2_index]) \
                     - (f1d[dim1_index](datum[dim1_index]) - f0) \
                     - (f1d[dim2_index](datum[dim2_index]) - f0) \
                     - f0
    return total


