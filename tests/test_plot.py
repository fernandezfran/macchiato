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

from macchiato.experiments.nmr import ChemicalShiftSpectra
from macchiato.experiments.xray import PairDistributionFunction

from matplotlib.testing.decorators import check_figures_equal

import numpy as np

import pytest

# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.parametrize(
    ("structure"),
    [("Li12Si7"), ("Li7Si3"), ("Li13Si4"), ("Li15Si4")],
)
class TestSpectraPlotter:
    """Test the ChemicalShiftCenters fitting."""

    @check_figures_equal(extensions=["png", "pdf"], tol=0.02)
    def test_versus_data(self, fig_test, fig_ref, structure, request):
        """Test the fit attributes of the ChemicalShiftCenters."""
        structure = request.getfixturevalue(structure)

        csc = structure["contributions"]
        csw = np.array(
            [structure["sigma"], structure["gamma"], structure["heigth"]]
        )

        # test plot
        test_ax = fig_test.subplots()
        ChemicalShiftSpectra(csc, csw).plot.versus_data(
            structure["ppm"], structure["intensity"], ax=test_ax
        )

        # ref plot
        ref_ax = fig_ref.subplots()

        ref_ax.scatter(structure["ppm"], structure["intensity"])

        ref_ax.plot(structure["ppm"], structure["ypred"])

        ref_ax.set_xlim((structure["ppm"].max(), structure["ppm"].min()))
        ref_ax.set_xlabel(r"$\delta$ [ppm]")


class TestPDFPlotter:
    """Test the ChemicalShiftCenters fitting."""

    @check_figures_equal(extensions=["png", "pdf"], tol=0.02)
    def test_versus_data(self, fig_test, fig_ref, gofr):
        pdf = PairDistributionFunction(gofr["universes"])
        X, y = gofr["pdf"]["r"], gofr["pdf"]["gofr"]
        pdf.fit(X, y)

        # test plot
        test_ax = fig_test.subplots()
        pdf.plot.versus_data(X, y, ax=test_ax)

        # ref plot
        ref_ax = fig_ref.subplots()

        ref_ax.scatter(X, y)
        ref_ax.plot(pdf.rbins_, gofr["pred"])

        ref_ax.set_xlim(X.min(), X.max())
        ref_ax.set_xlabel(r"r [$\AA$]")
        ref_ax.set_ylabel("G(r)")
