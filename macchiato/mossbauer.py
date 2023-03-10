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

"""Mössbauer Effect Spectroscopy."""

# ============================================================================
# IMPORTS
# ============================================================================

import MDAnalysis as mda

import numpy as np

from .base import NearestNeighbors
from .config import CONFIG

# ============================================================================
# CLASSES
# ============================================================================


class MossbauerEffect(NearestNeighbors):
    """Delta of a splitting in a two-contribution peak in Mossbauer Effect.

    Parameters
    ----------
    u : MDAnalysis.core.universe.Universe
        a universe with the box defined

    start : int, default=None
        start frame of analysis

    stop : int, default=None
        stop frame of analysis

    step : int, default=None
        number of frames to skip between each analyzed one

    Attributes
    ----------
    contributions_ : numpy.ndarray
        the mean of the Mössbauer effect delta between spectra peaks per
        `atom_type` atom
    """

    def __init__(self, u, start=None, stop=None, step=None):
        super().__init__(
            u,
            CONFIG["mossbauer"]["atom_type"],
            start=start,
            stop=stop,
            step=step,
        )

        self.all_atoms = u.select_atoms("all")

    def _contribution(self, lowest):
        """Contribution of each `atom_type` atom given the lowest value."""
        return CONFIG["mossbauer"]["contributions"][
            "mix" if lowest >= CONFIG["mossbauer"]["threshold"] else "unmixed"
        ]

    def _mean_contribution(self):
        """Mean contribution per atom to the delta between peaks."""
        all_distances = mda.lib.distances.distance_array(
            self.atom_group, self.all_atoms, box=self.u.dimensions
        )

        for i, distances in enumerate(all_distances):
            first_coordination_shell = np.where(
                distances < CONFIG["mossbauer"]["rcut"]
            )[0]

            conc = np.mean(
                [
                    self.all_atoms[neighbor].name
                    == CONFIG["mossbauer"]["atom_type"]
                    for neighbor in first_coordination_shell
                ]
            )
            lowest = min(conc, 1 - conc)

            self.contributions_[i] += self._contribution(lowest)

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
        """Compute the clustering and predict the delta splitting.

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
            the mean of the Mössbauer effect delta between spectra peaks per
            `atom_type` atom
        """
        return super().fit_predict(X, y, sample_weight)
