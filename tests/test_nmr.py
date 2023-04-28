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

from macchiato.experiments.nmr import (
    ChemicalShiftCenters,
    ChemicalShiftWidth,
)

import numpy as np

import pytest

# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.parametrize(
    ("structure"),
    [("Li12Si7"), ("Li7Si3"), ("Li13Si4"), ("Li15Si4")],
)
class TestChemicalShiftCenters:
    """Test the ChemicalShiftCenters fitting."""

    def test_fit(self, structure, request):
        """Test the fit attributes of the ChemicalShiftCenters."""
        structure = request.getfixturevalue(structure)

        csc = ChemicalShiftCenters(structure["u"])

        csc.fit(None)

        np.testing.assert_array_almost_equal(csc.bonded_, structure["bonded"])
        np.testing.assert_array_almost_equal(
            csc.isolated_, structure["isolated"]
        )

    def test_fit_predict(self, structure, request):
        """Test the fit_predict of the ChemicalShiftCenters."""
        structure = request.getfixturevalue(structure)

        csc = ChemicalShiftCenters(structure["u"])

        contributions = csc.fit_predict(None)

        np.testing.assert_array_almost_equal(
            contributions, structure["contributions"]
        )


@pytest.mark.parametrize(
    ("structure"),
    [("Li12Si7"), ("Li7Si3"), ("Li13Si4"), ("Li15Si4")],
)
class TestChemicalShiftWidth:
    """Test the ChemicalShiftWidth fitting."""

    def test_fit(self, structure, request):
        """Test the fit attributes of the ChemicalShiftWidth."""
        structure = request.getfixturevalue(structure)

        csw = ChemicalShiftWidth(structure["contributions"])
        csw.fit(structure["ppm"], structure["intensity"])

        np.testing.assert_array_almost_equal(csw.sigma_, structure["sigma"])
        np.testing.assert_array_almost_equal(csw.gamma_, structure["gamma"])
        np.testing.assert_array_almost_equal(csw.heigth_, structure["heigth"])

    def test_predict(self, structure, request):
        """Test the ChemicalShiftWidth predict."""
        structure = request.getfixturevalue(structure)

        csw = ChemicalShiftWidth(structure["contributions"])
        csw._popt = [
            structure["sigma"],
            structure["gamma"],
            structure["heigth"],
        ]

        ypred = csw.predict(structure["ppm"])

        np.testing.assert_array_almost_equal(ypred, structure["ypred"], 5)

    def test_score(self, structure, request):
        """Test the score of the ChemicalShiftWidth."""
        structure = request.getfixturevalue(structure)

        csw = ChemicalShiftWidth(structure["contributions"])
        csw._popt = [
            structure["sigma"],
            structure["gamma"],
            structure["heigth"],
        ]

        score = csw.score(structure["ppm"], structure["intensity"])

        np.testing.assert_array_almost_equal(score, structure["score"])
