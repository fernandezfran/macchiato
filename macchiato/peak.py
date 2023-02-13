#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import scipy.special


def voigt(x, mean, sigma, gamma, heigth=1.0):
    return heigth * np.real(
        scipy.special.wofz((x - mean + 1j * gamma) / sigma / np.sqrt(2))
        / (sigma * np.sqrt(2))
    )
