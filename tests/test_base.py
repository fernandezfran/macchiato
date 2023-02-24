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

from macchiato.base import NearestNeighbors

import pytest

# =============================================================================
# TESTS
# =============================================================================


def test_base_raise(Li12Si7):
    """Test the NotImplementedError."""
    with pytest.raises(NotImplementedError):
        NearestNeighbors(
            Li12Si7["u"], "Li", "Si", None, None
        )._mean_contribution(None, None)
