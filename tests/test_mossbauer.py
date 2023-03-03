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

from macchiato.mossbauer import MossbauerEffect

import numpy as np

import pytest

# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.parametrize(
    ("structure"),
    [("Li12Si7"), ("Li7Si3"), ("Li13Si4"), ("Li15Si4")],
)
class TestMossbauerEffect:
    """Test the MossbauerEffect class."""

    def test_fit_predict(self, structure, request):
        """Test the fit_predict of the MossbauerEffect."""
        structure = request.getfixturevalue(structure)

        csc = MossbauerEffect(
            structure["u"], "Si", 3.4, {"mix": 1.2, "unmixed": 0.4}
        )

        contributions = csc.fit_predict(None)

        np.testing.assert_array_almost_equal(
            contributions, structure["mossbauer_contributions"]
        )
