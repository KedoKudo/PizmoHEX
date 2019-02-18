#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for post-processing tomography reconsturction results.
"""

import numpy   as np

from scipy.signal       import medfilt2d
from PizmoHEX.npmath.cv import calc_histequal_wgt

__all__ = [
    "enhance_img",
]

def enhance_img(img, median_ks=3, normalized=True):
    """
    Enhance the image from aps 1ID to counter its weak contrast nature

    Parameters
    ----------
    img : ndarray
        original projection image collected at APS 1ID
    median_ks: int
        kernel size of the 2D median filter, must be odd
    normalized: bool, optional
        specify whether the enhanced image is normalized between 0 and 1,
        default is True

    Returns
    -------
    ndarray
        enhanced projection image
    """
    img =   calc_histequal_wgt(img) \
          * medfilt2d(img, kernel_size=median_ks).astype(np.float64)**2
    return img/img.max() if normalized else img
