#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for preprocessing tomography data collected with high-energy X-ray
(synchrotron radiation) source.
"""

import numpy              as np
import concurrent.futures as cf
import multiprocessing    as mp

from scipy.signal  import medfilt2d
from scipy.ndimage import gaussian_filter

__all__ = [
    "sino_normalize_background_aps_1id",
    "tomo_normalize_background_aps_1id",
    ]


def sino_normalize_background_aps_1id(sino, beam_is_moving=True):
    """
    Normalize the background (air) in each row of the sinogram (before
    minus_log)

    Parameters
    ----------
    sino: ndarray
        2D sinograms with the unit of counts
    beam_is_moving: bool
        If the beam is assumed to be moving (default Ture), linear 
        interpolation is required during background normalization

    Returns
    -------
    ndarray
        Sinogram with background normalized to 1 (air, no attenuation)
    """
    # amplify the variation in background
    sino = np.sqrt(sino)

    # use median filter and gaussian filter to locate the sample region
    # -- median filter is to counter impulse noise
    # -- gaussian filter is for estimating the sample location
    prof = np.gradient(np.sum(gaussian_filter(medfilt2d(sino,
                                                        kernel_size=3,
                                                        ),
                                              sigma=50,
                                             ),
                              axis=0,
                              )
                      )

    # find the left and right bound of the sample
    edgeLeft = max(prof.argmin(), 11)  #
    edgeRigth = min(prof.argmax(), sino.shape[1]-11)

    # locate the left and right background
    bgL = np.average(sino[:, 1:edgeLeft], axis=1)
    bgR = np.average(sino[:, edgeRigth:-1], axis=1)

    # calculate alpha
    alpha = np.ones_like(sino)
    # NOTE:
    #   If the beam is wobbling horizontally, it is necessary
    #   to perform the interpolation.
    #   Otherwise, simple average would sufice.
    if beam_is_moving:
        for n in range(alpha.shape[0]):
            alpha[n, :] = np.linspace(bgL[n], bgR[n], alpha.shape[1])
    else:
        alpha *= ((bgL+bgR)/2)[:, None]

    return (sino/alpha)**2


def tomo_normalize_background_aps_1id(tomo, 
                                      beam_is_moving=True, 
                                      ncore=None):
    """
    Normalize the background of the given tomo stack (before -log) to 
    one (air, no attenuation).

    Parameters
    ----------
    tomo: np.ndarray
        tomopy images stacks (axis_0 is the oemga direction, before -log)
    beam_is_moving: bool
        If the beam is assumed to be moving (default Ture), linear interpolation
        is required during background normalization
    ncore: int
        Number of cores to use

    Returns
    -------
    Tomo image stack with background normalized to 1
    """
    ncore = mp.cpu_count()-1 if ncore is None else ncore

    tmp = []
    with cf.ProcessPoolExecutor(ncore) as e:
        for n_sino in range(tomo.shape[1]):
            tmp.append(e.submit(sino_normalize_background_aps_1id,
                                tomo[:, n_sino, :],
                                beam_is_moving=beam_is_moving,
                                )
                       )
    return np.stack([me.result() for me in tmp], axis=1)