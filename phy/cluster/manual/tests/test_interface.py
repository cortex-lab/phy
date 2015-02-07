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
from ....io.mock.artificial import MockModel


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_interface():
    session = start_manual_clustering(model=MockModel())
    view = session.show_waveforms()
    session.select([0])
    view_bis = session.show_waveforms()
    session.merge([3, 4])

    view.close()
    view_bis.close()

    session = start_manual_clustering(model=MockModel())
    session.select([1, 2])
    view = session.show_waveforms()
    view.close()
