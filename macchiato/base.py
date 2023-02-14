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

"""FirstNeighbors base class."""

# ============================================================================
# IMPORTS
# ============================================================================

import exma

import numpy as np

import sklearn.cluster
from sklearn.base import BaseEstimator, ClusterMixin

# ============================================================================
# CLASSES
# ============================================================================


class FirstNeighbors(ClusterMixin, BaseEstimator):
    """First Neighbors Clustering.

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

    Attributes
    ----------
    bonded_ : float
        the percentage of bonded cluters of `cluster_type` atoms

    isolated_ : float
        the percentage of isolated `cluster_type` atoms

    contributions_ : np.array
        the contribution of the `atom_type` atoms
    """

    def __init__(
        self, trajectory, atom_type, cluster_type, rcut_atom, rcut_cluster
    ):
        self.trajectory = trajectory

        self.atom_type = atom_type
        self.cluster_type = cluster_type

        self.rcut_atom = rcut_atom
        self.rcut_cluster = rcut_cluster

        self._n_atom_type = trajectory[0]._natoms_type(
            trajectory[0]._mask_type(atom_type)
        )
        self._n_cluster_type = trajectory[0]._natoms_type(
            trajectory[0]._mask_type(cluster_type)
        )

        self.bonded_ = []
        self.isolated_ = []
        self.contributions_ = np.zeros(self._n_atom_type)

    def _isolated_or_bonded(self, snapshot):
        """Isolated/bonded `cluster_type` atoms per snapshot."""
        distance_matrix = exma.distances.pbc_distances(
            snapshot,
            snapshot,
            type_c=self.cluster_type,
            type_i=self.cluster_type,
        )

        db = sklearn.cluster.DBSCAN(
            eps=self.rcut_cluster, min_samples=2, metric="precomputed"
        )
        db.fit(distance_matrix)

        nisol = np.count_nonzero(db.labels_ == -1)

        self.isolated_.append(nisol)
        self.bonded_.append(self._n_cluster_type - nisol)

        return db.labels_

    def _mean_contribution(self, snapshot, labels):
        """Mean contribution of the `atom_type` atoms."""
        raise NotImplementedError

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
        for snapshot in self.trajectory:
            self._mean_contribution(
                snapshot, self._isolated_or_bonded(snapshot)
            )

        self.bonded_ = np.mean(self.bonded_) / self._n_cluster_type
        self.isolated_ = np.mean(self.isolated_) / self._n_cluster_type

        self.contributions_ /= len(self.trajectory)

        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        """Compute the clustering and predict the contributions.

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
            the contribution of the `atom_type` atoms
        """
        self.fit(X)
        return self.contributions_
