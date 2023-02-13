#!/usr/bin/env python
# -*- coding: utf-8 -*-

import importlib_metadata

from .chemical_shift import ChemicalShiftSpectra
from .peak import voigt

__all__ = ["ChemicalShiftSpectra", "voigt"]

NAME = "macchiato"

DOC = __doc__

VERSION = importlib_metadata.version(NAME)

__version__ = tuple(VERSION.split("."))

del importlib_metadata
