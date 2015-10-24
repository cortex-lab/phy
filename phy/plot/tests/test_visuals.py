# -*- coding: utf-8 -*-

"""Test visuals."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from pytest import mark

from ..visuals import ScatterVisual, PlotVisual, HistogramVisual


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------


def _test_visual(qtbot, c, v, stop=False, **kwargs):
    v.attach(c)
    v.set_data(**kwargs)
    c.show()
    if stop:  # pragma: no cover
        qtbot.stop()


#------------------------------------------------------------------------------
# Test scatter visual
#------------------------------------------------------------------------------

def test_scatter_empty(qtbot, canvas):
    pos = np.zeros((0, 2))
    _test_visual(qtbot, canvas, ScatterVisual(), pos=pos)


@mark.parametrize('marker_type', ScatterVisual._supported_marker_types)
def test_scatter_markers(qtbot, canvas_pz, marker_type):
    n = 100
    pos = .2 * np.random.randn(n, 2)
    _test_visual(qtbot, canvas_pz,
                 ScatterVisual(marker_type=marker_type),
                 pos=pos)


def test_scatter_custom(qtbot, canvas_pz):

    n = 100

    # Random position.
    pos = .2 * np.random.randn(n, 2)

    # Random colors.
    c = np.random.uniform(.4, .7, size=(n, 4))
    c[:, -1] = .5

    # Random sizes
    s = 5 + 20 * np.random.rand(n)

    _test_visual(qtbot, canvas_pz, ScatterVisual(),
                 pos=pos, colors=c, size=s)


#------------------------------------------------------------------------------
# Test plot visual
#------------------------------------------------------------------------------

def test_plot_empty(qtbot, canvas):
    data = np.zeros((1, 0))
    _test_visual(qtbot, canvas, PlotVisual(),
                 data=data)


def test_plot_0(qtbot, canvas_pz):
    data = np.zeros((1, 10))
    _test_visual(qtbot, canvas_pz, PlotVisual(),
                 data=data)


def test_plot_1(qtbot, canvas_pz):
    data = .2 * np.random.randn(1, 10)
    _test_visual(qtbot, canvas_pz, PlotVisual(),
                 data=data)


def test_plot_2(qtbot, canvas_pz):

    n_signals = 50
    data = 20 * np.random.randn(n_signals, 10)

    # Signal bounds.
    b = np.zeros((n_signals, 4))
    b[:, 0] = -1
    b[:, 1] = np.linspace(-1, 1 - 2. / n_signals, n_signals)
    b[:, 2] = 1
    b[:, 3] = np.linspace(-1 + 2. / n_signals, 1., n_signals)

    # Signal colors.
    c = np.random.uniform(.5, 1, size=(n_signals, 4))
    c[:, 3] = .5

    _test_visual(qtbot, canvas_pz, PlotVisual(),
                 data=data, data_bounds=[-10, 10],
                 signal_bounds=b, signal_colors=c)


#------------------------------------------------------------------------------
# Test histogram visual
#------------------------------------------------------------------------------

def test_histogram_empty(qtbot, canvas):
    hist = np.zeros((1, 0))
    _test_visual(qtbot, canvas, HistogramVisual(),
                 hist=hist)


def test_histogram_0(qtbot, canvas_pz):
    hist = np.zeros((1, 10))
    _test_visual(qtbot, canvas_pz, HistogramVisual(),
                 hist=hist)


def test_histogram_1(qtbot, canvas_pz):
    hist = np.random.rand(1, 10)
    _test_visual(qtbot, canvas_pz, HistogramVisual(),
                 hist=hist)


def test_histogram_2(qtbot, canvas_pz):

    n_hists = 5
    hist = np.random.rand(n_hists, 21)

    # Histogram colors.
    c = np.random.uniform(.3, .6, size=(n_hists, 4))
    c[:, 3] = 1

    _test_visual(qtbot, canvas_pz, HistogramVisual(),
                 hist=hist, hist_colors=c, hist_lims=2 * np.ones(n_hists))
