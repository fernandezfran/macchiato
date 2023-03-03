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

import numpy as np

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

    def __init__(self, u, atom_type, start=None, stop=None, step=None):
        self.u = u

        self.atom_group = u.select_atoms(f"name {atom_type}")
        self._n_atoms_type = len(self.atom_group)

        self.start = 0 if start is None else start
        self.stop = len(u.trajectory) if stop is None else stop
        self.step = 1 if step is None else step

        self.bonded_ = []
        self.isolated_ = []
        self.contributions_ = np.zeros(self._n_atoms_type)

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
        raise NotImplementedError

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
