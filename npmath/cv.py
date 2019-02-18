#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for computer vision (image processing) related functions.
"""

import numpy as np

from scipy.signal  import medfilt2d

__all__ = [
    "selective_median_filter",
]


def selective_median_filter(img, threshold=None, kernel_size=3):
    """
    Use selective median filtering to remove impulse noises

    Parameters
    ----------
    img : ndarray
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
