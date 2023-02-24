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

"""Simple nearest-neighbor model to atom cluster types."""

# ============================================================================
# IMPORTS
# ============================================================================

import importlib_metadata

from .chemical_shift import (
    ChemicalShiftCenters,
    ChemicalShiftSpectra,
    ChemicalShiftWidth,
)

# ============================================================================
# CONSTANTS
# ============================================================================

__all__ = [
    "ChemicalShiftCenters",
    "ChemicalShiftWidth",
    "ChemicalShiftSpectra",
]

NAME = "macchiato"

DOC = __doc__

VERSION = importlib_metadata.version(NAME)

__version__ = tuple(VERSION.split("."))

del importlib_metadata
