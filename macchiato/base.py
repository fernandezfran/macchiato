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

"""NearestNeighbors base class."""

# ============================================================================
# IMPORTS
# ============================================================================

import MDAnalysis as mda

import numpy as np

import sklearn.cluster
from sklearn.base import BaseEstimator, ClusterMixin

# ============================================================================
# CLASSES
# ============================================================================


class NearestNeighbors(ClusterMixin, BaseEstimator):
    """Nearest Neighbors Clustering.

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

    contributions_ : np.array
        the contribution of the `atom_type` atoms
    """

    def __init__(
        self,
        u,
        atom_type,
        cluster_type,
        rcut_atom,
        rcut_cluster,
        start=None,
        stop=None,
        step=None,
    ):
        self.u = u

        self.atom_group = self.u.select_atoms(f"name {atom_type}")
        self.cluster_group = self.u.select_atoms(f"name {cluster_type}")

        self._n_atoms_type = len(self.atom_group)
        self._n_cluster_type = len(self.cluster_group)

        self.rcut_atom = rcut_atom
        self.rcut_cluster = rcut_cluster

        self.start = 0 if start is None else start
        self.stop = len(u.trajectory) if stop is None else stop
        self.step = 1 if step is None else step

        self.bonded_ = []
        self.isolated_ = []
        self.contributions_ = np.zeros(self._n_atoms_type)

    def _isolated_or_bonded(self, cluster_distances):
        """Isolated/bonded `cluster_type` atoms per snapshot."""
        db = sklearn.cluster.DBSCAN(
            eps=self.rcut_cluster, min_samples=2, metric="precomputed"
        )
        db.fit(cluster_distances)

        nisol = np.count_nonzero(db.labels_ == -1)

        self.isolated_.append(nisol)
        self.bonded_.append(self._n_cluster_type - nisol)

        return db.labels_

    def _mean_contribution(self, atom_to_cluster_distances, labels):
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
        for i, ts in enumerate(self.u.trajectory):
            if i < self.start:
                continue

            if i % self.step == 0:
                cluster_dist = mda.lib.distances.distance_array(
                    self.cluster_group,
                    self.cluster_group,
                    box=self.u.dimensions,
                )
                labels = self._isolated_or_bonded(cluster_dist)

                atom_to_cluster_dist = mda.lib.distances.distance_array(
                    self.atom_group, self.cluster_group, box=self.u.dimensions
                )
                self._mean_contribution(atom_to_cluster_dist, labels)

            if i >= self.stop:
                break

        self.bonded_ = np.mean(self.bonded_) / self._n_cluster_type
        self.isolated_ = np.mean(self.isolated_) / self._n_cluster_type

        self.contributions_ *= self.step / (self.stop - self.start)

        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        """Compute the clustering and predict the contributions.

        Parameters
        ----------
        X : ignored
            not used here, just convention, it uses the snapshots in the
            xyz_fname

        y : ignored
            not used, just convention

        Returns
        -------
        contributions_ : np.array
            the contribution of the `atom_type` atoms
        """
        self.fit(X)
        return self.contributions_
