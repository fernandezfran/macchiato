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

"""Config of models."""

# ============================================================================
# IMPORTS
# ============================================================================

import os
import pathlib

import yaml

# ============================================================================
# CONSTANTS
# ============================================================================

MACCHIATO_PATH = pathlib.Path(
    os.path.join(os.path.abspath(os.path.dirname(__file__)))
)

with open(MACCHIATO_PATH / "config.yml", "r") as cfg:
    CONFIG = yaml.load(cfg, Loader=yaml.FullLoader)
