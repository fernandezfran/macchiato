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

import exma

import numpy as np

import scipy.optimize

from sklearn.base import RegressorMixin

from .base import FirstNeighbors
from .plot import CSPlotter
from .utils import voigt_peak

# ============================================================================
# CLASSES
# ============================================================================


class ChemicalShiftCenters(FirstNeighbors):
    """Obtain the peak centers of each atom `atom_type` in a trajectory.

    Parameters
    ----------
    trajectory : `exma.core.AtomicSystem` iterable
        a molecular dynamics trajectory with the box defined

    atom_type : str or int
        type of atom on which to analyze the proximity to the clusters

    cluster_type : str or int
        type of atom forming the clusters

    rcut_atom : float
        cutoff radius of first-neighbor of atoms `atom_type` to `cluster_type`
        ones

    rcut_cluster : float
        cutoff radius to consider a cluster of `cluter_type` atoms

    ppm : dict
        dictionary with two keys `bonded` and `isolated` with the contribution
        to the chemical shift spectra of each class

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
        self, trajectory, atom_type, cluster_type, rcut_atom, rcut_cluster, ppm
    ):
        super().__init__(
            trajectory, atom_type, cluster_type, rcut_atom, rcut_cluster
        )

        self.ppm = ppm

    def _mean_contribution(self, snapshot, labels):
        """Mean contribution per atom to the chemical shift spectra."""
        atom_to_cluster_matrix = exma.distances.pbc_distances(
            snapshot,
            snapshot,
            type_c=self.atom_type,
            type_i=self.cluster_type,
        )

        for k, distances in enumerate(atom_to_cluster_matrix):
            contribution = 0
            index = np.where(distances < self.rcut_atom)[0]
            for idx in index:
                contribution += (
                    self.ppm["isolated"]
                    if labels[idx] == -1
                    else self.ppm["bonded"]
                )

            self.contributions_[k] += contribution / index.size

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
        return super().fit(X, y, sample_weight)

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

    def _nmr_profile(self, X, sigma, gamma, heigth):
        """NMR profile with a contribution per center."""
        return np.mean(
            [
                voigt_peak(X, mean, sigma, gamma, heigth=heigth)
                for mean in self.csc
            ],
            axis=0,
        ).ravel()

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
            if isinstance(csc, np.ndarray)
            else np.array([csw.sigma_, csw.gamma_, csw.heigth_])
        )

    @property
    def plot(self):
        """Plot accesor to macchiato.plot.CSPlotter."""
        return CSPlotter(self)
