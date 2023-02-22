#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This file is part of macchiato
#   https://github.com/fernandezfran/macchiato/
# Copyright (c) 2023, Francisco Fernandez
# License: MIT
#   https://github.com/fernandezfran/macchiato/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""NMR profile peaks."""

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np

import scipy.special

# ============================================================================
# FUNCTIONS
# ============================================================================


def voigt_peak(x, mean, sigma, gamma, heigth=1.0):
    """Voigt profile.

    Parameters
    ----------
    x : np.ndarray
        x values to evaluate the voigt peak

    mean : float
        expected value or mean

    sigma : float
        the standard deviation of the gaussian component

    gamma : float
        the half-width at half-maximum of the lorentzian component

    heigth : float, default=1.0
        optional heigth of the peak

    Returns
    -------
    np.ndarray
        evaluation of the voigt peak in the x values given the parameters
    """
    return (
        heigth
        * np.real(
            scipy.special.wofz((x - mean + 1j * gamma) / sigma / np.sqrt(2))
        )
        / sigma
        / np.sqrt(2 * np.pi)
    )


def nmr_profile(X, centers, sigma, gamma, heigth=1.0):
    """NMR profile with a voigt contribution per center.

    Parameters
    ----------
    X : np.ndarray
        X values to evaluate the voigt peak

    centers : np.ndarray
        array with expected value or mean per center

    sigma : float
        the standard deviation of the gaussian component

    gamma : float
        the half-width at half-maximum of the lorentzian component

    heigth : float, default=1.0
        optional heigth of the peak

    Returns
    -------
    np.ndarray
        evaluation of the voigt peak in the x values given the parameters
    """
    return np.mean(
        [voigt_peak(X, mean, sigma, gamma, heigth=heigth) for mean in centers],
        axis=0,
    ).ravel()
