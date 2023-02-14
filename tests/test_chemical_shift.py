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

from macchiato.chemical_shift import ChemicalShiftSpectra

import numpy as np

import pytest

# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.parametrize(
    ("structure"),
    [
        ("Li13Si64"),
        ("Li36Si64"),
        ("Li57Si64"),
        ("Li128Si64"),
        ("Li210Si64"),
        ("Li240Si64"),
    ],
)
def test_fit(structure, request):
    """Test the fit attributes of the ChemicalShiftSpectra."""
    structure = request.getfixturevalue(structure)

    css = ChemicalShiftSpectra(
        structure["trajectory"],
        "Li",
        "Si",
        3.4,
        3.0,
        {"bonded": 18.0, "isolated": 6.0},
    )

    css.fit(None)

    np.testing.assert_array_almost_equal(css.bonded_, structure["bonded"])
    np.testing.assert_array_almost_equal(css.isolated_, structure["isolated"])


@pytest.mark.parametrize(
    ("structure"),
    [
        ("Li13Si64"),
        ("Li36Si64"),
        ("Li57Si64"),
        ("Li128Si64"),
        ("Li210Si64"),
        ("Li240Si64"),
    ],
)
def test_fit_predict(structure, request):
    """Test the fit_predict of the ChemicalShiftSpectra."""
    structure = request.getfixturevalue(structure)

    css = ChemicalShiftSpectra(
        structure["trajectory"],
        "Li",
        "Si",
        3.4,
        3.0,
        {"bonded": 18.0, "isolated": 6.0},
    )

    contributions = css.fit_predict(None)

    np.testing.assert_array_almost_equal(
        contributions, structure["contributions"]
    )
