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

"""Estimate the peak centers of each Li atom in a Si environment and the \
overall width of the chemical shift spectra."""

# ============================================================================
# IMPORTS
# ============================================================================

import MDAnalysis as mda

import numpy as np

import scipy.optimize

import sklearn.cluster
from sklearn.base import RegressorMixin

from .base import NearestNeighbors
from .config import CONFIG
from .plot import SpectraPlotter
from .utils import nmr_profile

# ============================================================================
# CLASSES
# ============================================================================


class ChemicalShiftCenters(NearestNeighbors):
    r"""Obtain the peak centers of each Li atom.

    The nearest-neighbor model consist of considering that each Li atom makes
    a particular contribution to the total chemical shift spectra. The ansatz
    chosen to locate the center of the peak was inspired by a comment from
    Ket et al. [1]_, who stated that if a Li atom is near a bonded Si, then
    the center of its peak is at 18ppm, whereas if it is near an isolated Si
    atom, the center should be at 6ppm. To take into account intermediate
    contributions that appears in the spectra we define the peak position of
    each Li atom as follow,

    :math:`\delta_{Li} = \frac{1}{N_{Si}} \sum_{Si \in NN} \delta_{Key}`,

    where the sum is considered over all first nearest-neighbors (NN) Si atoms
    (:math:`N_{Si}`) and :math:`\delta_{Key}` is the shift value of 18ppm or
    6ppm depending of the Si type.

    Parameters
    ----------
    u : MDAnalysis.core.universe.Universe
        a universe with the box defined

    start : int, default=None
        start frame of analysis

    stop : int, default=None
        stop frame of analysis

    step : int, default=None
        number of frames to skip between each analyzed one

    Attributes
    ----------
    bonded_ : float
        the percentage of bonded Si atoms

    isolated_ : float
        the percentage of isolated Si atoms

    contributions_ : numpy.ndarray
        the mean of the peak in the chemical shift spectra per Li atom

    References
    ----------
    .. [1] Key, Baris, et al. "Real-time NMR investigations of structural
        changes in silicon electrodes for lithium-ion batteries." `Journal of
        the American Chemical Society` 131.26 (2009): 9239-9249.
    """

    def __init__(self, u, start=None, stop=None, step=None):
        self.cluster_group = u.select_atoms(
            f"name {CONFIG['chemical_shift']['cluster_type']}"
        )

        self._n_cluster_type = len(self.cluster_group)

        super().__init__(
            u,
            CONFIG["chemical_shift"]["atom_type"],
            start=start,
            stop=stop,
            step=step,
        )

        self.bonded_, self.isolated_ = [], []

    def _isolated_or_bonded(self):
        """Isolated/bonded Si atoms per snapshot."""
        cluster_distances = mda.lib.distances.distance_array(
            self.cluster_group, self.cluster_group, box=self.u.dimensions
        )
        db = sklearn.cluster.DBSCAN(
            eps=CONFIG["chemical_shift"]["rcut_cluster"],
            min_samples=2,
            metric="precomputed",
        )
        db.fit(cluster_distances)

        nisol = np.count_nonzero(db.labels_ == -1)

        self.isolated_.append(nisol)
        self.bonded_.append(self._n_cluster_type - nisol)

        return db.labels_

    def _mean_contribution(self):
        """Mean contribution per Li atom to the chemical shift spectra."""
        labels = self._isolated_or_bonded()

        atom_to_cluster_dist = mda.lib.distances.distance_array(
            self.atom_group, self.cluster_group, box=self.u.dimensions
        )

        for i, distances in enumerate(atom_to_cluster_dist):
            first_coordination_shell = np.where(
                distances < CONFIG["chemical_shift"]["rcut_atom"]
            )[0]

            self.contributions_[i] += np.mean(
                [
                    CONFIG["chemical_shift"]["contributions"][
                        "isolated" if labels[neighbor] == -1 else "bonded"
                    ]
                    for neighbor in first_coordination_shell
                ]
            )

    def fit(self, X, y=None, sample_weight=None):
        """Fit method.

        Parameters
        ----------
        X : ignored
            not used here, just convention, it uses the snapshots in the
            trajectory

        y : ignored
            not used, just convention

        Returns
        -------
        self : object
            fitted model
        """
        super().fit(X, y, sample_weight)

        self.bonded_ = np.mean(self.bonded_) / self._n_cluster_type
        self.isolated_ = np.mean(self.isolated_) / self._n_cluster_type

        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        """Compute the clustering and predict the Li chemical shift centers.

        Parameters
        ----------
        X : ignored
            not used here, just convention, it uses the snapshots in the
            trajectory

        y : ignored
            not used, just convention

        Returns
        -------
        contributions_ : numpy.ndarray
            the mean of the peak in the chemical shift spectra per Li atom
        """
        return super().fit_predict(X, y, sample_weight)


class ChemicalShiftWidth(RegressorMixin):
    """Fit the overall width of a Li chemical shift spectra given the centers.

    A Voigt contribution is assumed for each center with the same width and
    same heigth and the total intensity is a sum of them.

    Parameters
    ----------
    csc : macchiato.chemical_shift.ChemicalShiftCenters or numpy.ndarray
        a ChemicalShiftCenters object already fitted or a numpy array with the
        centers

    Attributes
    ----------
    sigma_ : float
        the fitted standard deviation of the gaussian component of each voigt
        peak

    gamma_ : float
        the fitted half-width at half-maximum of the lorentzian component of
        each voigt peak

    heigth_ : float
        the fitted heigth of each voigt peak
    """

    def __init__(self, csc):
        self.csc = csc if isinstance(csc, np.ndarray) else csc.contributions_

        self._nmr_profile = lambda X, sigma, gamma, heigth: nmr_profile(
            X, self.csc, sigma, gamma, heigth
        )

    def fit(self, X, y):
        """Fit the width of the Li NMR profile to the experimental data.

        Parameters
        ----------
        X : array-like of shape (n_ppm, 1)
            chemical shift ppm points

        y : array-like of shape (n_ppm,)
            target intensity

        Returns
        -------
        self : object
            fitted peaks
        """
        self._popt, _ = scipy.optimize.curve_fit(self._nmr_profile, X, y)

        self.sigma_, self.gamma_, self.heigth_ = self._popt

        return self

    def predict(self, X):
        """Predict the Li chemical shift spectra.

        Parameters
        ----------
        X : array-like of shape (n_ppm, 1)
            chemical shift ppm points

        Returns
        -------
        y : array-like of shape (n_ppm,)
            predicted intensity
        """
        return self._nmr_profile(X, *self._popt)

    def score(self, X, y):
        """Return the coefficient of determination of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_ppm, 1)
            chemical shift ppm points

        y : array-like of shape (n_ppm,)
            true intensity

        Returns
        -------
        score : float
            :math:`R^2` of ``self.predict(X)`` wrt. `y`.
        """
        return super(ChemicalShiftWidth, self).score(X, y)


class ChemicalShiftSpectra:
    """Plot the Li chemical shift spectra once you have the centers and width.

    Parameters
    ----------
    csc : macchiato.chemical_shift.ChemicalShiftCenters or numpy.ndarray
        a ChemicalShiftCenters object already fitted or a numpy array with the
        centers

    csw : macchiato.chemical_shift.ChemicalShiftWidth or numpy.ndarray
        a ChemicalShiftWidth object already fitted or numpy.ndarray with
        sigma, gamma and heigth params of each voigt peak per Li atom
    """

    def __init__(self, csc, csw):
        self.centers = (
            csc if isinstance(csc, np.ndarray) else csc.contributions_
        )
        self.voigt_params = (
            csw
            if isinstance(csw, np.ndarray)
            else np.array([csw.sigma_, csw.gamma_, csw.heigth_])
        )

    @property
    def plot(self):
        """Plot accesor to :ref:`macchiato.plot` SpectraPlotter."""
        return SpectraPlotter(self)
