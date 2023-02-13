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

"""Simple first-neighbor model to atom clusters."""

# ============================================================================
# IMPORTS
# ============================================================================

import importlib_metadata

from .chemical_shift import ChemicalShiftSpectra
from .peak import voigt

# ============================================================================
# CONSTANTS
# ============================================================================

__all__ = ["ChemicalShiftSpectra", "voigt"]

NAME = "macchiato"

DOC = __doc__

VERSION = importlib_metadata.version(NAME)

__version__ = tuple(VERSION.split("."))

del importlib_metadata
