# -*- coding: utf-8 -*-

"""Test CCG plotting."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ...utils.logging import set_level
from ..traces import TraceView
from ...utils._color import _random_color
from ...io.mock.artificial import artificial_traces
from ...utils.testing import show_test


#------------------------------------------------------------------------------
# Tests VisPy
#------------------------------------------------------------------------------

def _test_traces(n_samples=None):
    n_channels = 20

    traces = artificial_traces(n_samples, n_channels)

    c = TraceView()
    c.visual.traces = traces

    show_test(c)


def test_traces_empty():
    _test_traces(n_samples=0)


def test_traces_full():
    _test_traces(n_samples=1000)
