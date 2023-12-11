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

import sklearn.metrics

from ..config import CONFIG
from ..plot import PDFPlotter

# ============================================================================
# CLASSES
# ============================================================================


class PairDistributionFunction:
    r"""X-ray Pair Distribution Function (PDF or :math:`G(r)`).

    PDF can be computed from Radial Distribution Function (RDF) by considering
    the contribution of each interaccion (Si-Si, Si-Li, Si-Si) given its
    scattering factor. Then, a measurement that has a mixture of alloys can be
    fitted to determine the weight factor of each one to predict the
    experiment.

    Parameters
    ----------
    universes : list of MDAnalysis.core.universe.Universe
        a list of universes with the box defined per alloy to be considered

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

    def _calculate_gofrs(self):
        """Calculate the gofr of each universe."""
        for universe in self.universes:
            if len(set(universe.atoms.types)) == 1:
                scattering_factors = (1,)
                interactions = (["all", "all"],)
            else:
                scattering_factors = self._cfg["scattering_factors"]
                interactions = it.combinations_with_replacement(
                    self._cfg["atom_types"], 2
                )

            gofr = np.zeros(self._cfg["nbins"])

            for factor, types in zip(scattering_factors, interactions):
                central = universe.select_atoms(types[0])
                interact = universe.select_atoms(types[1])

                rdf = mda_rdf.InterRDF(
                    central,
                    interact,
                    nbins=self._cfg["nbins"],
                    range=self._cfg["range"],
                    exclusion_block=(1, 1),
                )
                rdf.run()

                gofr += factor * rdf.results.rdf

            rbins = rdf.results.bins

            # volume of orthorhombic box
            volume = np.mean(
                [np.prod(universe.dimensions[:3]) for _ in universe.trajectory]
            )
            natoms = len(universe.atoms)
            rho = natoms / volume

            self.gofrs_.append(4 * np.pi * rho * rbins * (gofr - 1))

        self.rbins_ = rbins
        self.gofrs_ = np.asarray(self.gofrs_)

    def _objective(self, target, contributions, params):
        """Objective function to minimize."""
        return sklearn.metrics.mean_squared_error(
            target, params[-1] + np.dot(np.array(params[:-1]), contributions)
        )

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
        self._calculate_gofrs()

        # interpolate experimental data to the bins
        X = X.ravel()
        experiment = scipy.interpolate.interp1d(X, y)

        # fit the weights of each alloy
        min_mask = self.rbins_ > X.min()
        max_mask = self.rbins_ <= self._cfg["rmax"]
        mask = min_mask & max_mask

        rbins = self.rbins_[mask]
        contributions = np.array([gofr[mask] for gofr in self.gofrs_])

        target = experiment(rbins)
        params0 = np.ones(len(contributions) + 1) / len(contributions)
        bounds = [(0, None)] * len(contributions) + [(None, None)]
        results = scipy.optimize.minimize(
            lambda params: self._objective(target, contributions, params),
            params0,
            method="L-BFGS-B",
            bounds=bounds,
        )

        self.weights_ = np.array(results.x[:-1])
        self.offset_ = results.x[-1]

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
        return self.offset_ + np.dot(self.weights_, self.gofrs_)

    @property
    def plot(self):
        """Plot accesor to :ref:`macchiato.plot` PDFPlotter."""
        return PDFPlotter(self)
