# -*- coding: utf-8 -*-

"""Test plotting/VisPy utilities."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os
import os.path as op

import numpy as np
from numpy.testing import assert_allclose as ac
from vispy import config

from phy.electrode.mea import linear_positions
from ..utils import (_load_shader,
                     _tesselate_histogram,
                     _enable_depth_mask,
                     _boxes_overlap,
                     _get_boxes,
                     )


#------------------------------------------------------------------------------
# Test utilities
#------------------------------------------------------------------------------

def test_load_shader():
    assert 'main()' in _load_shader('simple.vert')
    assert config['include_path']
    assert op.exists(config['include_path'][0])
    assert op.isdir(config['include_path'][0])
    assert os.listdir(config['include_path'][0])


def test_tesselate_histogram():
    n = 7
    hist = np.arange(n)
    thist = _tesselate_histogram(hist)
    assert thist.shape == (6 * n, 2)
    ac(thist[0], [0, 0])
    ac(thist[-3], [n, n - 1])
    ac(thist[-1], [n, 0])


def test_enable_depth_mask(qtbot, canvas):

    @canvas.connect
    def on_draw(e):
        _enable_depth_mask()

    canvas.show()
    qtbot.waitForWindowShown(canvas.native)


def test_boxes_overlap():

    def _get_args(boxes):
        x0, y0, x1, y1 = np.array(boxes).T
        x0 = x0[:, np.newaxis]
        x1 = x1[:, np.newaxis]
        y0 = y0[:, np.newaxis]
        y1 = y1[:, np.newaxis]
        return x0, y0, x1, y1

    boxes = [[-1, -1, 0, 0], [0.01, 0.01, 1, 1]]
    x0, y0, x1, y1 = _get_args(boxes)
    assert not _boxes_overlap(x0, y0, x1, y1)

    boxes = [[-1, -1, 0.1, 0.1], [0, 0, 1, 1]]
    x0, y0, x1, y1 = _get_args(boxes)
    assert _boxes_overlap(x0, y0, x1, y1)


def test_get_boxes():
    positions = [[-1, -1], [1., 1.]]
    x0, y0, x1, y1 = _get_boxes(positions)
    assert np.all(x1 - x0 >= .4)
    assert np.all(y1 - y0 >= .4)
    assert not _boxes_overlap(x0, y0, x1, y1)

    positions = linear_positions(4)
    x0, y0, x1, y1 = _get_boxes(positions)
    assert not _boxes_overlap(x0, y0, x1, y1)
