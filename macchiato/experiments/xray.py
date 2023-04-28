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

"""Predictions of PDF by fitting experiments to the structures."""

# ============================================================================
# IMPORTS
# ============================================================================

import itertools as it

import MDAnalysis.analysis.rdf as mda_rdf

import numpy as np

import scipy.interpolate
import scipy.optimize

from ..base import NearestNeighbors
from ..config import CONFIG

# ============================================================================
# CLASSES
# ============================================================================


class PairDistributionFunction(NearestNeighbors):
    r"""X-ray Pair Distribution Function (PDF or :math:`G(r)`).

    PDF can be computed from Radial Distribution Function (RDF) by considering
    the contribution of each interaccion (Si-Si, Si-Li, Si-Si). Then, a
    measurement that has a mixture of alloys can be fitted to determine the
    weight factor of each one to predict the experiment.

    Parameters
    ----------
    universes : list of MDAnalysis.core.universe.Universe
        a universe with the box defined per alloy to be considered

    Attributes
    ----------
    weights_ : numpy.ndarray
        weight of each alloy in the same order as in the list of universes

    offset_ : float
        y-axis offset

    rbins_ : numpy.ndarray
        rvalues corresponding to the gofr y-values

    gofrs_ : list of numpy.ndarray
        a list with the PDF of each alloy in the same order as the list of
        universes
    """

    def __init__(self, universes):
        self.universes = universes

        self.weights_, self.offset_ = None, None
        self.gofrs_ = []

        self._cfg = CONFIG["pdf"]

    def fit(self, X, y):
        """Fit the weights of each alloy.

        Parameters
        ----------
        X : array-like of shape (rvalues, 1)
            r values

        y : array-like of shape (rvalues,)
            target intensity of the total PDF

        Returns
        -------
        self : object
            fitted weights
        """
        # first compute all the gofrs
        for u in self.universes:
            if len(set(u.atoms.types)) == 1:
                weights = (1,)
                interactions = (["all", "all"],)
            else:
                weights = self._cfg["weights"]
                interactions = it.combinations_with_replacement(
                    self._cfg["atom_types"], 2
                )

            gofr = np.zeros(self._cfg["nbins"])

            for w, types in zip(weights, interactions):
                central = u.select_atoms(types[0])
                interact = u.select_atoms(types[1])

                rdf = mda_rdf.InterRDF(
                    central,
                    interact,
                    nbins=self._cfg["nbins"],
                    range=self._cfg["range"],
                    exclusion_block=(1, 1),
                )
                rdf.run()

                gofr += w * rdf.results.rdf

            r = rdf.results.bins

            # volume of orthorhombic box
            volume = np.mean(
                [np.prod(u.dimensions[:3]) for ts in u.trajectory]
            )
            natoms = len(u.atoms)
            rho = natoms / volume

            self.gofrs_.append(4 * np.pi * rho * r * (gofr - 1))

        # interpolate experimental data to the bins
        X = X.ravel()
        experiment = scipy.interpolate.interp1d(X, y)

        self.rbins_ = r

        min_mask = r > X.min()
        max_mask = r <= self._cfg["rmax"]
        mask = min_mask & max_mask

        r = r[mask]
        target = experiment(r)

        contributions = [gofr[mask] for gofr in self.gofrs_]

        # fit the weights of each alloy
        def objective_function(params):
            return np.sum(
                (
                    np.sum(
                        [p * c for p, c in zip(params[:-1], contributions)],
                        axis=0,
                    )
                    + params[-1]
                    - target
                )
                ** 2
            )

        params0 = np.ones(len(contributions) + 1) / len(contributions)
        bounds = [(0, None)] * len(contributions) + [(None, None)]
        results = scipy.optimize.minimize(
            objective_function, params0, method="L-BFGS-B", bounds=bounds
        )
        params = results.x

        self.weights_ = params[:-1]
        self.offset_ = params[-1]

    def predict(self, X):
        """Predict the X-ray PDF.

        Parameters
        ----------
        X : array-like of shape (rvalues, 1)
            rvalues

        Returns
        -------
        y : array-like of shape (rvalues,)
            predicted intensity of the total PDF
        """
        return self.offset_ + np.sum(
            [w * gofr for w, gofr in zip(self.weights_, self.gofrs_)], axis=0
        )

    def fit_predict(self, X, y):
        """Fit and predict the X-ray PDF.

        Parameters
        ----------
        X : array-like of shape (rvalues, 1)
            r values

        y : array-like of shape (rvalues,)
            target intensity of the total PDF

        Returns
        -------
        y : array-like of shape (rvalues,)
            predicted intensity of the total PDF
        """
        return self.fit(X, y).predict(X)
