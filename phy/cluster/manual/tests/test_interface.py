# -*- coding: utf-8 -*-

"""Tests of manual clustering interface."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os

import numpy as np
from numpy.testing import assert_array_equal as ae
from pytest import raises

from ..interface import start_manual_clustering
from ....datasets.mock import MockModel


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_interface():
    session = start_manual_clustering(model=MockModel())
    assert session
