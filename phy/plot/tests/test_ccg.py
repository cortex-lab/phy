# -*- coding: utf-8 -*-

"""Test CCG plotting."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import mark

import numpy as np

from ..ccg import _plot_ccg_mpl, CorrelogramView, plot_correlograms
from ...utils._color import _random_color
from ...io.mock import artificial_correlograms
from ...utils.testing import show_test


# Skip these tests in "make test-quick".
pytestmark = mark.long()


#------------------------------------------------------------------------------
# Tests matplotlib
#------------------------------------------------------------------------------

def test_plot_ccg():
    n_bins = 51
    ccg = np.random.randint(size=n_bins, low=10, high=50)
    _plot_ccg_mpl(ccg, baseline=20, color='g')


def test_plot_correlograms():
    n_bins = 51
    ccg = np.random.uniform(size=(3, 3, n_bins))
    c = plot_correlograms(ccg, lines=[-10, 0, 20], show=False)
    show_test(c)


#------------------------------------------------------------------------------
# Tests VisPy
#------------------------------------------------------------------------------

def _test_correlograms(n_clusters=None):
    n_samples = 51

    correlograms = artificial_correlograms(n_clusters, n_samples)

    c = CorrelogramView(keys='interactive')
    c.cluster_ids = np.arange(n_clusters)
    c.visual.correlograms = correlograms
    c.visual.cluster_colors = np.array([_random_color()
                                        for _ in range(n_clusters)])
    c.lines = [-5, 0, 5]
    show_test(c)


def test_correlograms_empty():
    _test_correlograms(n_clusters=0)


def test_correlograms_full():
    _test_correlograms(n_clusters=3)
