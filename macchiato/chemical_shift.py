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

"""Estimate the peak centers of each atom and the overall width of the \
chemical shift spectra."""

# ============================================================================
# IMPORTS
# ============================================================================

import MDAnalysis as mda

import numpy as np

import scipy.optimize

import sklearn.cluster
from sklearn.base import RegressorMixin

from .base import NearestNeighbors
from .plot import SpectraPlotter
from .utils import nmr_profile

# ============================================================================
# CLASSES
# ============================================================================


class ChemicalShiftCenters(NearestNeighbors):
    """Obtain the peak centers of each atom `atom_type` in a trajectory.

    Parameters
    ----------
    u : MDAnalysis.core.universe.Universe
        a universe with the box defined

    atom_type : str or int
        type of atom on which to analyze the proximity to the clusters

    cluster_type : str or int
        type of atom forming the clusters

    rcut_atom : float
        cutoff radius of first coordination shell of atoms `atom_type` to
        `cluster_type` ones

    rcut_cluster : float
        cutoff radius to consider a cluster of `cluter_type` atoms

    ppm : dict
        dictionary with two keys `bonded` and `isolated` with the contribution
        to the chemical shift spectra of each class

    start : int, default=None
        start frame of analysis

    stop : int, default=None
        stop frame of analysis

    step : int, default=None
        number of frames to skip between each analyzed one

    Attributes
    ----------
    bonded_ : float
        the percentage of bonded cluters of `cluster_type` atoms

    isolated_ : float
        the percentage of isolated `cluster_type` atoms

    contributions_ : numpy.ndarray
        the mean of the peak in the chemical shift spectra per atom of the
        `atom_type` type
    """

    def __init__(
        self,
        u,
        atom_type,
        cluster_type,
        rcut_atom,
        rcut_cluster,
        ppm,
        start=None,
        stop=None,
        step=None,
    ):
        self.cluster_group = u.select_atoms(f"name {cluster_type}")

        self._n_cluster_type = len(self.cluster_group)

        self.rcut_atom = rcut_atom
        self.rcut_cluster = rcut_cluster

        super().__init__(u, atom_type, start=start, stop=stop, step=step)

        self.bonded_, self.isolated_ = [], []

        self.ppm = ppm

    def _isolated_or_bonded(self):
        """Isolated/bonded `cluster_type` atoms per snapshot."""
        cluster_distances = mda.lib.distances.distance_array(
            self.cluster_group,
            self.cluster_group,
            box=self.u.dimensions,
        )
        db = sklearn.cluster.DBSCAN(
            eps=self.rcut_cluster, min_samples=2, metric="precomputed"
        )
        db.fit(cluster_distances)

        nisol = np.count_nonzero(db.labels_ == -1)

        self.isolated_.append(nisol)
        self.bonded_.append(self._n_cluster_type - nisol)

        return db.labels_

    def _mean_contribution(self):
        """Mean contribution per atom to the chemical shift spectra."""
        labels = self._isolated_or_bonded()

        atom_to_cluster_dist = mda.lib.distances.distance_array(
            self.atom_group, self.cluster_group, box=self.u.dimensions
        )
        for i, distances in enumerate(atom_to_cluster_dist):
            first_coordination_shell = np.where(distances < self.rcut_atom)[0]

            self.contributions_[i] += np.mean(
                [
                    self.ppm[
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
        """Compute the clustering and predict the chemical shift centers.

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
            the chemical shift center per atom of the `atom_type` atoms
        """
        return super().fit_predict(X, y, sample_weight)


class ChemicalShiftWidth(RegressorMixin):
    """Fit the overall widht of a chemical shift spectra given the centers.

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
        """Fit the width of the nmr profile to the experimental data.

        X : array-like of shape (n_ppm, 1)
            chemical shift ppm points

        y : array-like of shape (n_ppm,)
            target intensity

        Returns
        -------
        self : object
            fitted widths
        """
        self._popt, _ = scipy.optimize.curve_fit(self._nmr_profile, X, y)

        self.sigma_, self.gamma_, self.heigth_ = self._popt

        return self

    def predict(self, X):
        """Predict the chemical shift spectra.

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
    """Plot the chemical shift spectra once you have the centers and width.

    Parameters
    ----------
    csc : macchiato.chemical_shift.ChemicalShiftCenters or numpy.ndarray
        a ChemicalShiftCenters object already fitted or a numpy array with the
        centers

    csw : macchiato.chemical_shift.ChemicalShiftWidth or numpy.ndarray
        a ChemicalShiftWidth object already fitted or numpy.ndarray with
        sigma, gamma and heigth params of the voigt peak
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
