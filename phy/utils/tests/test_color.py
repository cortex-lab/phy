# -*- coding: utf-8 -*-

"""Test colors."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import colorcet as cc
import numpy as np
from numpy.testing import assert_almost_equal as ae

from pytest import raises

from ..color import (
    _is_bright, _random_bright_color, spike_colors, add_alpha, selected_cluster_color,
    _override_hsv, _hex_to_triplet, _continuous_colormap, _categorical_colormap,
    _selected_cluster_idx, ClusterColorSelector, _add_selected_clusters_colors)


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_random_color():
    for _ in range(20):
        assert _is_bright(_random_bright_color())


def test_hex_to_triplet():
    assert _hex_to_triplet('#0123ab')


def test_add_alpha():
    assert add_alpha((0, .5, 1), .75) == (0, .5, 1, .75)
    assert add_alpha(np.random.rand(5, 3), .5).shape == (5, 4)

    assert add_alpha((0, .5, 1, .1), .75) == (0, .5, 1, .75)
    assert add_alpha(np.random.rand(5, 4), .5).shape == (5, 4)


def test_override_hsv():
    assert _override_hsv((.1, .9, .5), h=1, s=0, v=1) == (1, 1, 1)


def test_selected_cluster_color():
    c = selected_cluster_color(0)
    assert isinstance(c, tuple)
    assert len(c) == 4


def test_colormaps():
    colormap = np.array(cc.glasbey_bw_minc_20_minl_30)
    values = np.random.randint(10, 20, size=100)
    colors = _categorical_colormap(colormap, values)
    assert colors.shape == (100, 3)

    colormap = np.array(cc.rainbow_bgyr_35_85_c73)
    values = np.linspace(0, 1, 100)
    colors = _continuous_colormap(colormap, values)
    assert colors.shape == (100, 3)


def test_spike_colors():
    spike_clusters = [1, 0, 0, 3]
    cluster_ids = [0, 1, 2, 3]
    colors = spike_colors(spike_clusters, cluster_ids)
    assert colors.shape == (4, 4)
    ae(colors[1], colors[2])


def test_cluster_color_selector_1():
    cluster_ids = [1, 2, 3]
    c = ClusterColorSelector(lambda cid: cid * .1, cluster_ids=cluster_ids)

    assert len(c.get(1, alpha=.5)) == 4
    ae(c.get_values([0, 0]), np.zeros(2))

    for colormap in ('linear', 'rainbow', 'categorical', 'diverging'):
        c.set_color_mapping(colormap=colormap)
        colors = c.get_colors(cluster_ids)
        assert colors.shape == (3, 4)


def test_cluster_color_selector_2():
    cluster_ids = [2, 3, 5, 7]
    c = ClusterColorSelector(
        lambda cid: cid, cluster_ids=cluster_ids, colormap='categorical', categorical=True)

    c2 = c.get_colors([2])
    c3 = c.get_colors([3])
    c7 = c.get_colors([7])
    c2_ = c.get_colors([2])
    ae(c2, c2_)

    c.set_cluster_ids([3, 7, 11])
    c3_ = c.get_colors([3])
    c7_ = c.get_colors([7])
    ae(c3, c3_)
    ae(c7, c7_)

    with raises(AssertionError):
        ae(c2, c3)


def test_cluster_color_group():
    # Mock ClusterMeta instance, with 'fields' property and get(field, cluster) function.
    cluster_ids = [1, 2, 3]
    c = ClusterColorSelector(
        lambda cl: {1: None, 2: 'mua', 3: 'good'}[cl], cluster_ids=cluster_ids)

    c.set_color_mapping(colormap='cluster_group')
    colors = c.get_colors(cluster_ids)
    assert colors.shape == (3, 4)


def test_cluster_color_log():
    cluster_ids = [1, 2, 3]
    c = ClusterColorSelector(lambda cid: cid * .1, cluster_ids=cluster_ids)

    c.set_color_mapping(logarithmic=True)
    colors = c.get_colors(cluster_ids)
    assert colors.shape == (3, 4)


def test_add_selected_clusters_colors_1():
    cluster_colors = np.tile(np.c_[np.arange(3)], (1, 3))
    cluster_colors = add_alpha(cluster_colors)
    cluster_colors_sel = _add_selected_clusters_colors([1], [0, 1, 3], cluster_colors)
    ae(cluster_colors_sel[[0]], add_alpha(np.zeros((1, 3))))
    ae(cluster_colors_sel[[2]], add_alpha(2 * np.ones((1, 3))))
    # Cluster at index 0 is selected, should be in blue.
    r, g, b, _ = cluster_colors_sel[1]
    assert b > g > r


def test_add_selected_clusters_colors_2():
    selected_clusters, cluster_ids = [5, 3, 2], [12, 7, 5, 2, 1]
    clu_idx, cmap_idx = _selected_cluster_idx(selected_clusters, cluster_ids)
    ae(clu_idx, [2, 3])
    ae(cmap_idx, [0, 2])

    cluster_colors = np.c_[np.arange(5), np.zeros((5, 3))]
    cluster_colors_sel = _add_selected_clusters_colors(
        selected_clusters, cluster_ids, cluster_colors)

    ae(cluster_colors_sel[[0, 1, 4]], cluster_colors[[0, 1, 4]])
    ae(cluster_colors_sel[2], selected_cluster_color(0))
    ae(cluster_colors_sel[3], selected_cluster_color(2))
