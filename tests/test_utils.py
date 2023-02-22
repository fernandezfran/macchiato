#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This file is part of macchiato
#   https://github.com/fernandezfran/macchiato/
# Copyright (c) 2023, Francisco Fernandez
# License: MIT
#   https://github.com/fernandezfran/macchiato/blob/master/LICENSE

# ============================================================================
# IMPORTS
# ============================================================================

from macchiato.utils import nmr_profile, voigt_peak

import numpy as np

# =============================================================================
# TESTS
# =============================================================================


def test_nmr_profile():
    """Test the NMR profile."""
    ref = np.array(
        [6.9583, 0.0193, 0.009, 0.0196, 6.9591, 0.0196, 0.009, 0.0193, 6.9583]
    )

    x = np.arange(-1, 1.01, 0.25)
    y = nmr_profile(x, np.array([-1, 0, 1]), 0.01, 0.01)

    np.testing.assert_array_almost_equal(y, ref, 4)


def test_voigt_peak():
    """Test the voigt profile peak."""
    ref = np.array(
        [
            3.18374e-03,
            5.66086e-03,
            1.27426e-02,
            5.10933e-02,
            2.08709e01,
            5.10933e-02,
            1.27426e-02,
            5.66086e-03,
            3.18374e-03,
        ]
    )

    x = np.arange(-1, 1.01, 0.25)
    y = voigt_peak(x, 0, 0.01, 0.01)

    np.testing.assert_array_almost_equal(y, ref, 4)
