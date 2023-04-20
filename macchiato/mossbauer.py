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

"""Predict Mössbauer 119Sn (equiv. Si atoms) spectra peaks splitting."""

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
    r"""Delta of a splitting in a two-contribution peak in Mössbauer spectra.

    We define a :math:`z`-value as the minimum between Li and Si
    concentrations in the first coordination shell of each Si atom, i.e.,
    :math:`0 \leq z \leq 0.5`. Then, there is a linear relationship between
    :math:`\Delta` of each Si atom with its :math:`z`-value. This was inspired
    by Li et al. work [2]_.

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
        the mean of the Mössbauer effect delta between spectra peaks per Si
        atom

    References
    ----------
    .. [2] Li, Jing, et al. "In situ 119Sn Mössbauer effect study of the
        reaction of lithium with Si using a Sn probe." `Journal of The
        Electrochemical Society` 156.4 (2009): A283.
    """

    def __init__(self, u, start=None, stop=None, step=None):
        self._cfg = CONFIG["mossbauer"]

        super().__init__(
            u, self._cfg["atom_type"], start=start, stop=stop, step=step
        )

        self.all_atoms = u.select_atoms("all")

    def _contribution(self, lowest):
        """Contribution of each Si atom given the lowest value."""
        return (
            self._cfg["contributions"]["intercept"]
            + self._cfg["contributions"]["slope"] * lowest
        )

    def _mean_contribution(self):
        """Mean contribution per Si atom to the delta between peaks."""
        all_distances = mda.lib.distances.distance_array(
            self.atom_group, self.all_atoms, box=self.u.dimensions
        )

        for i, distances in enumerate(all_distances):
            first_coordination_shell = np.where(distances < self._cfg["rcut"])[
                0
            ]

            conc = np.mean(
                [
                    self.all_atoms[neighbor].name == self._cfg["atom_type"]
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
        r"""Compute the clustering and predict the delta splitting.

        To obtain each :math:`\Delta` an average is perfomed in the trajectory
        of the Si atom.

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
            the mean of the Mössbauer effect delta between spectra peaks per Si
            atom
        """
        return super().fit_predict(X, y, sample_weight)
