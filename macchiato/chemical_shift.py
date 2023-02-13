#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import exma

import numpy as np

from .base import FirstNeighbors


class ChemicalShiftSpectra(FirstNeighbors):
    def __init__(
        self, trajectory, atom_type, cluster_type, rcut_atom, rcut_cluster, ppm
    ):
        super().__init__(
            trajectory, atom_type, cluster_type, rcut_atom, rcut_cluster
        )

        self.ppm = ppm

    def _mean_contribution(self, snapshot, labels):
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
        return super().fit(X, y, sample_weight)

    def fit_predict(self, X, y=None, sample_weight=None):
        return super().fit_predict(X, y, sample_weight)
