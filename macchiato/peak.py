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

"""Voigt profile peak."""

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np

import scipy.special

# ============================================================================
# FUNCTIONS
# ============================================================================


def voigt(x, mean, sigma, gamma, heigth=1.0):
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

    height : float, default=1.0
        optional height of the peak

    Returns
    -------
    np.ndarray
        evaluation of the voigt peak in the x values given the parameters
    """
    return heigth * np.real(
        scipy.special.wofz((x - mean + 1j * gamma) / sigma / np.sqrt(2))
        / (sigma * np.sqrt(2))
    )
