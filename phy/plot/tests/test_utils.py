# -*- coding: utf-8 -*-

"""Test plotting/VisPy utilities."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os
import os.path as op

import numpy as np
from numpy.testing import assert_array_equal as ae
from numpy.testing import assert_allclose as ac
from vispy import config

from phy.electrode.mea import linear_positions, staggered_positions
from ..utils import (_load_shader,
                     _tesselate_histogram,
                     _enable_depth_mask,
                     _get_data_bounds,
                     _boxes_overlap,
                     _binary_search,
                     _get_boxes,
                     _get_box_pos_size,
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


def test_get_data_bounds():
    db0 = np.array([[0, 1, 4, 5],
                    [0, 1, 4, 5],
                    [0, 1, 4, 5]])
    arr = np.arange(6).reshape((3, 2))
    assert np.all(_get_data_bounds(None, arr) == [[0, 1, 4, 5]])

    db = db0.copy()
    assert np.all(_get_data_bounds(db, arr) == [[0, 1, 4, 5]])

    db = db0.copy()
    db[2, :] = [1, 1, 1, 1]
    assert np.all(_get_data_bounds(db, arr)[:2, :] == [[0, 1, 4, 5]])
    assert np.all(_get_data_bounds(db, arr)[2, :] == [0, 0, 2, 2])

    db = db0.copy()
    db[:2, :] = [1, 1, 1, 1]
    assert np.all(_get_data_bounds(db, arr)[:2, :] == [[0, 0, 2, 2]])
    assert np.all(_get_data_bounds(db, arr)[2, :] == [0, 1, 4, 5])


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


def test_binary_search():
    def f(x):
        return x < .4
    ac(_binary_search(f, 0, 1), .4)
    ac(_binary_search(f, 0, .3), .3)
    ac(_binary_search(f, .5, 1), .5)


def test_get_boxes():
    positions = [[-1, 0], [1, 0]]
    boxes = _get_boxes(positions)
    ac(boxes, [[-1, -.25, 0, .25],
               [+0, -.25, 1, .25]], atol=1e-4)

    positions = [[-1, 0], [1, 0]]
    boxes = _get_boxes(positions, keep_aspect_ratio=False)
    ac(boxes, [[-1, -1, 0, 1],
               [0, -1, 1, 1]], atol=1e-4)

    positions = linear_positions(4)
    boxes = _get_boxes(positions)
    ac(boxes, [[-0.5, -1.0, +0.5, -0.5],
               [-0.5, -0.5, +0.5, +0.0],
               [-0.5, +0.0, +0.5, +0.5],
               [-0.5, +0.5, +0.5, +1.0],
               ], atol=1e-4)

    positions = staggered_positions(8)
    boxes = _get_boxes(positions)
    ac(boxes[:, 1], np.arange(.75, -1.1, -.25), atol=1e-6)
    ac(boxes[:, 3], np.arange(1, -.76, -.25), atol=1e-7)


def test_get_box_pos_size():
    bounds = [[-1, -.25, 0, .25],
              [+0, -.25, 1, .25]]
    pos, size = _get_box_pos_size(bounds)
    ae(pos, [[-.5, 0], [.5, 0]])
    assert size == (.5, .25)
