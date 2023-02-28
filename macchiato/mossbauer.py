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

"""Mössbauer Effect."""

# ============================================================================
# IMPORTS
# ============================================================================

from .base import NearestNeighbors

# ============================================================================
# CLASSES
# ============================================================================


class MossbauerEffect(NearestNeighbors):
    """Mossbauer Effect.

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
        start=None,
        stop=None,
        step=None,
    ):
        super().__init__(
            u,
            atom_type,
            cluster_type,
            rcut_atom,
            rcut_cluster,
            start=start,
            stop=stop,
            step=step,
        )

    def _mean_contribution(self, atom_to_cluster_distances, labels):
        """To be implemented."""
        raise NotImplementedError

    def fit(self, X, y=None, sample_weight=None):
        """Fit method.

        To be implemented.
        """
        raise NotImplementedError

    def fit_predict(self, X, y=None, sample_weight=None):
        """To be implemented."""
        return super().fit_predict(X, y, sample_weight)
