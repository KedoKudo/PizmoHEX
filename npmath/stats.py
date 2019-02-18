#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for statistics related functions.
"""

import numpy as np
import scipy as sp

__all__ = [
    "discrete_cdf",
    "gauss1d",
]

def gauss1d(data_1d, *p):
    """
    1D Gaussian function used for curve fitting.

    Parameters
    ----------
    data_1d:  np.ndarray
        1D array for curve fitting
    p:  parameter lis t
        magnitude, center, std = p

    Returns
    -------
    1d Gaussian distribution evaluted at data_1d with p
    """
    amplitude, mean, sigma = p
    return amplitude * np.exp(-(data_1d - mean)**2 / (2. * sigma**2))


def discrete_cdf(data, steps=None):
    """
    Calculate CDF of given data without discrete binning to avoid unnecessary
    skew of distribution.

    The default steps (None) will use the whole data. In other words, it is
    close to considering using bin_size=1 or bins=len(data).

    Parameters
    ----------
    data  :  np.ndarray
        1-D numpy array
    steps :  [ None | int ], optional
        Number of elements in the returning array

    Returns
    -------
    pltX  : np.ndarray
        Data along x (data) direction
    pltY  : np.ndarray
        Data along y (density) direction
    """
    data = np.sort(data)

    # check if list is empty
    if data.shape[0] == 0:
        return [], []

    # subsamping if steps is specified and the number is smaller than the
    # total lenght of x
    if (steps is not None) and len(data) > steps:
        data = data[np.arange(0, len(data), int(np.ceil(len(data) / steps)))]

    # calculate the cumulative density
    data_plt = np.tile(data, (2, 1)).flatten(order='F')
    density = np.arange(len(data))
    density_plt = np.vstack((density, density + 1)).flatten(order='F') / float(density[-1])

    return data_plt, density_plt


def calc_affine_transform(pts_source, pts_target):
    """
    Use least square regression to calculate the 2D affine transformation
    matrix (3x3, rot&trans) based on given set of (marker) points.
                            pts_source -> pts_target

    Parameters
    ----------
    pts_source  :  np.2darray
        source points with dimension of (n, 2) where n is the number of
        marker points
    pts_target  :  np.2darray
        target points where
                F(pts_source) = pts_target
    Returns
    -------
    np.2darray
        A 3x3 2D affine transformation matrix
          | r_11  r_12  tx |    | x1 x2 ...xn |   | x1' x2' ...xn' |
          | r_21  r_22  ty | *  | y1 y2 ...yn | = | y1' y2' ...yn' |
          |  0     0     1 |    |  1  1 ... 1 |   |  1   1  ... 1  |
        where r_ij represents the rotation and t_k represents the translation
    """
    # augment data with padding to include translation
    def pad(data_1d):
        return np.hstack([data_1d, np.ones((data_1d.shape[0], 1))])

    # NOTE:
    #   scipy affine_transform performs as np.dot(m, vec),
    #   therefore we need to transpose the matrix here
    #   to get the correct rotation

    return sp.linalg.lstsq(pad(pts_source), pad(pts_target))[0].T


def val_at_percentage(array, percentage):
    """
    Find the value at the given percentage in the histogram of a the given
    nd-array

    Parameters
    ----------
    array: ndarray
        data source
    percentage: float
        percentage of interest
    
    Returns
    -------
    float
        value at given percentage in the histogram of input array
    """
    return np.sort(array.flatten())[int(np.prod(array.shape)*percentage)]
