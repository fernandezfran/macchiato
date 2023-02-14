#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This file is part of macchiato
#   https://github.com/fernandezfran/macchiato/
# Copyright (c) 2023, Francisco Fernandez
# License: MIT
#   https://github.com/fernandezfran/macchiato/blob/master/LICENSE

# ============================================================================
# IMPORTS
# ============================================================================

import os
import pathlib

import exma

import numpy as np

import pytest

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture()
def data_path():
    return pathlib.Path(
        os.path.join(os.path.abspath(os.path.dirname(__file__)), "test_data")
    )


@pytest.fixture()
def Li13Si64(data_path):
    traj = exma.read_xyz(data_path / "Li13Si64.xyz")
    traj[0].box = np.full(3, 11.510442)
    return {
        "trajectory": traj,
        "bonded": 1.0,
        "isolated": 0.0,
        "contributions": np.full(13, 18.0),
    }
