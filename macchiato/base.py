#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import exma

import numpy as np

import sklearn.cluster
from sklearn.base import BaseEstimator, ClusterMixin


class FirstNeighbors(ClusterMixin, BaseEstimator):
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
        raise NotImplementedError

    def fit(self, X, y=None, sample_weight=None):
        for snapshot in self.trajectory:
            self._mean_contribution(
                snapshot, self._isolated_or_bonded(snapshot)
            )

        self.bonded_ = np.mean(self.bonded_) / self._n_cluster_type
        self.isolated_ = np.mean(self.isolated_) / self._n_cluster_type

        self.contributions_ /= len(self.trajectory)

        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        self.fit(X)
        return self.contributions_
