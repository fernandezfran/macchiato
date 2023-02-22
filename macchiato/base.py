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

import MDAnalysis as mda

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
    xyz_fname : str
        a string with the path of the xyz file with the trajectory snapshots

    boxes : iterable
        iterable with np.ndarray containing the box size
        [lx, ly, lz, alpha, beta, gamma].

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
        self,
        xyz_fname,
        boxes,
        atom_type,
        cluster_type,
        rcut_atom,
        rcut_cluster,
    ):
        self.u = mda.Universe(xyz_fname)

        self.boxes = boxes

        self.atom_group = self.u.select_atoms(f"name {atom_type}")
        self.cluster_group = self.u.select_atoms(f"name {cluster_type}")

        self._n_atoms_type = len(self.atom_group)
        self._n_cluster_type = len(self.cluster_group)

        self.rcut_atom = rcut_atom
        self.rcut_cluster = rcut_cluster

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
        for ts, box in zip(self.u.trajectory, self.boxes):
            cluster_dist = mda.lib.distances.distance_array(
                self.cluster_group, self.cluster_group, box=box
            )
            labels = self._isolated_or_bonded(cluster_dist)

            atom_to_cluster_dist = mda.lib.distances.distance_array(
                self.atom_group, self.cluster_group, box=box
            )
            self._mean_contribution(atom_to_cluster_dist, labels)

        self.bonded_ = np.mean(self.bonded_) / self._n_cluster_type
        self.isolated_ = np.mean(self.isolated_) / self._n_cluster_type

        self.contributions_ /= len(self.u.trajectory)

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
