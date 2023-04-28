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

from macchiato.experiments.xray import PairDistributionFunction

import numpy as np

# =============================================================================
# TESTS
# =============================================================================


class TestPairDistributionFunction:
    """Test the PairDistributionFunction class."""

    def test_fit(self, gofr):
        """Test the fit of the PairDistributionFunction."""
        pdf = PairDistributionFunction(gofr["universes"])

        X, y = gofr["pdf"]["r"], gofr["pdf"]["gofr"]
        pdf.fit(X, y)

        np.testing.assert_array_almost_equal(pdf.weights_, gofr["weights"])
        np.testing.assert_array_almost_equal(pdf.offset_, gofr["offset"])

    def test_predict(self, gofr):
        """Test the prediction of the PairDistributionFunction."""
        pdf = PairDistributionFunction(gofr["universes"])

        X, y = gofr["pdf"]["r"], gofr["pdf"]["gofr"]
        pdf.fit(X, y)

        pred = pdf.predict(pdf.rbins_)

        np.testing.assert_array_almost_equal(pred, gofr["pred"], decimal=5)
