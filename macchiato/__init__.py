#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of macchiato
#   https://github.com/fernandezfran/macchiato/
# Copyright (c) 2023, Francisco Fernandez
# License: MIT
#   https://github.com/fernandezfran/macchiato/blob/master/LICENSE

# ============================================================================
# DOCS
# ============================================================================

"""Data-driven nearest-neighbors models to predict physical experiments \
in silicon-based lithium-ion battery anodes."""

# ============================================================================
# IMPORTS
# ============================================================================

import importlib_metadata

from . import experiments
from .experiments.mossbauer import MossbauerEffect
from .experiments.nmr import (
    ChemicalShiftCenters,
    ChemicalShiftSpectra,
    ChemicalShiftWidth,
)
from .experiments.xray import PairDistributionFunction

# ============================================================================
# CONSTANTS
# ============================================================================

__all__ = [
    "experiments",
    "MossbauerEffect",
    "ChemicalShiftCenters",
    "ChemicalShiftWidth",
    "ChemicalShiftSpectra",
    "PairDistributionFunction",
]

NAME = "macchiato"

DOC = __doc__

VERSION = importlib_metadata.version(NAME)

__version__ = tuple(VERSION.split("."))

del importlib_metadata
