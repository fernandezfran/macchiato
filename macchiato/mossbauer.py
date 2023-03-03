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

import MDAnalysis as mda

import numpy as np

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
        type of atom to be analyzed

    rcut : float
        cutoff radius of first coordination shell of atoms `atom_type` to the
        rest

    mossbauer : dict
        dictionary with two keys `mix` and `unmixed` whit the contribution to
        the splitting of the two peaks in the Mössbauer effect spectroscopy

    threshold : float, default=0.25
        float between 0 and 1 that indicates the percentage from which there is
        a mix, for the default case, e.g. when there is 25% of the element with
        the lowest concentration it is considered that there is a mix

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

    def __init__(
        self,
        u,
        atom_type,
        rcut,
        mossbauer,
        threshold=0.25,
        start=None,
        stop=None,
        step=None,
    ):
        super().__init__(u, atom_type, start=start, stop=stop, step=step)

        self.atom_type = atom_type
        self.all_atoms = u.select_atoms("all")

        self.rcut = rcut

        self.mossbauer = mossbauer
        self.threshold = threshold

    def _mean_contribution(self):
        """Mean contribution per atom to the delta between peaks."""
        all_distances = mda.lib.distances.distance_array(
            self.atom_group, self.all_atoms, box=self.u.dimensions
        )

        for i, distances in enumerate(all_distances):
            first_coordination_shell = np.where(distances < self.rcut)[0]

            conc = np.mean(
                [
                    self.all_atoms[neighbor].name == self.atom_type
                    for neighbor in first_coordination_shell
                ]
            )
            lowest = min(conc, 1 - conc)

            self.contributions_[i] += np.mean(
                [
                    self.mossbauer[
                        "mix" if lowest >= self.threshold else "unmixed"
                    ]
                ]
            )

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
