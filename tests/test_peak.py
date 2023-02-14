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

from macchiato.peak import voigt

import numpy as np

# =============================================================================
# TESTS
# =============================================================================


def test_voigt():
    """Test the voigt profile peak."""
    ref = np.array(
        [
            5.643025e-03,
            1.003361e-02,
            2.258566e-02,
            9.056059e-02,
            3.699276e01,
            9.056059e-02,
            2.258566e-02,
            1.003361e-02,
            5.643025e-03,
        ]
    )

    x = np.arange(-1, 1.01, 0.25)
    y = voigt(x, 0, 0.01, 0.01)

    np.testing.assert_array_almost_equal(y, ref, 5)
