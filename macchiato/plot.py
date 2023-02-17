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

"""Plotter class."""

# ============================================================================
# IMPORTS
# ============================================================================

import matplotlib.pyplot as plt

import numpy as np

from .utils import voigt_peak

# ============================================================================
# CLASSES
# ============================================================================


class CSPlotter:
    """Chemical shift plotter.

    Kind of plots to produce:

    - 'spectra' : the predicted spectra
    - 'versus_data' : the predicted versus true spectra

    Parameters
    ----------
    css : macchiato.chemical_shift.ChemicalShiftSpectra
        an already fitted chemical shift spectra
    """

    def __init__(self, css):
        self.css = css

    def spectra(self, X, ax=None, **kwargs):
        """Plot the predicted spectra.

        Parameters
        ----------
        X : array-like of shape (n_ppm, 1)
            chemical shift ppm points to evaluate the spectra

        ax : matplotlib.axes.Axes, default=None
            the current axes

        **kwargs
            additional keyword arguments that are passed and are documented in
            ``matplotlib.axes.Axes.plot`` for the predictions values.

        Returns
        -------
        ax : matplotlib.axes.Axes
            the current axes
        """
        ax = plt.gca() if ax is None else ax

        y = np.mean(
            [
                voigt_peak(
                    X,
                    c,
                    *self.css.voigt_params,
                )
                for c in self.css.centers
            ],
            axis=0,
        ).ravel()

        ax.plot(X, y, **kwargs)

        ax.set_xlim((X.max(), X.min()))
        ax.set_xlabel(r"$\delta$ [ppm]")

        return ax

    def versus_data(self, X, y, ax=None, true_kws=None, pred_kws=None):
        """Plot the predicted spectra against the true data.

        Parameters
        ----------
        X : array-like of shape (n_ppm, 1)
            chemical shift ppm points

        y : array-like of shape (n_ppm,)
            true intensity

        ax : matplotlib.axes.Axes, default=None
            the current axes

        true_kws : dict, default=None
            additional keyword arguments that are passed and are documented in
            ``matplotlib.axes.Axes.scatter`` for the true values.

        pred_kws : dict, default=None
            additional keyword arguments that are passed and are documented in
            ``matplotlib.axes.Axes.plot`` for the predictions values.

        Returns
        -------
        ax : matplotlib.axes.Axes
            the current axes
        """
        ax = plt.gca() if ax is None else ax

        true_kws = {} if true_kws is None else true_kws
        true_kws.setdefault("label", "measured spectra")

        pred_kws = {} if pred_kws is None else pred_kws
        pred_kws.setdefault("label", "predicted spectra")

        ax.scatter(X, y, **true_kws)

        self.spectra(X, ax=ax, **pred_kws)

        return ax
