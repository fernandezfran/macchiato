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

from .base import FirstNeighbors

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

    contributions_ : np.array
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
        contributions_ : np.array
            the chemical shift center per atom of the `atom_type` atoms
        """
        return super().fit_predict(X, y, sample_weight)
