#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for computer vision (image processing) related functions.
"""

import numpy   as np

from scipy.signal  import medfilt2d

__all__ = [
    "calc_histequal_wgt",
    "selective_median_filter",
]


def selective_median_filter(img, threshold=None, kernel_size=3):
    """
    Use selective median filtering to remove impulse noises

    Parameters
    ----------
    img: ndarray
        2D images
    threshold: float
        Expected difference between impulse noise and the meidan
        value, default is 5% of the median value of the image
    kernel_size: int
        Kernel size of the median filter

    Returns
    -------
    ndarray
        Image with impulse noises replaced with the meidan values.
    """
    threshold = 0.05*np.median(img) if threshold is None else threshold

    _img_median = medfilt2d(img, kernel_size=kernel_size)

    return np.where(np.absolute(img - _img_median) < threshold,
                    img,
                    _img_median,
                   )


def calc_histequal_wgt(img):
    """
    Calculate the histogram equalization weight for a given image

    Parameters
    ----------
    img: ndarray
        2D images

    Returns
    -------
    ndarray
        histogram euqalization weights (0-1) in the same shape as original
        image
    """
    return (np.sort(img.flatten()).searchsorted(img) + 1)/np.prod(img.shape)

def svd_enhance(img, eigen_cut=None):
    """
    Reduce noise and compress image using Singular Value Decomposition

    Parameters
    ----------
    img:  ndarray
        2D images
    eigen_cut: int
        the number of primary eigen features retained in returning image

    Returns
    ndarray
        Image after the SVD denosing/compression.
    -------
    """
    __U, __S, __V = np.linalg.svd(img, full_matrices=True)
    # NOTE:
    # Use the strongest 80% eigen features by default
    if eigen_cut is None:
        eigen_cut = int(0.8*min(__U.shape[1], __V.shape[0]))  
    else:
        eigen_cut = eigen_cut
    return np.dot(__U[:,:eigen_cut]*__S[:eigen_cut], __V[:eigen_cut,:])


def rescale_image(img, minval=0, maxval=1):
    '''
    Rescale givem image intesnity between 0 and 1 through linear
    interpolcation.

    Parameters
    ----------
    img: ndarray
        2D image for rescaling
    minval: float
        Intensity lower bound after rescaling
    maxval: float
        Intensity upper bound fater rescaling

    Returns
    -------
    ndarray
        Rescaled image
    '''
    img = np.array(img)
    img = img - img.min() + minval
    img = img/(img.max()) 
    return img * maxval
