# -*- coding: utf-8 -*-

"""Plotting CCGs."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from vispy import gloo

from ._mpl_utils import _bottom_left_frame
from ._vispy_utils import (BaseSpikeVisual,
                           BaseSpikeCanvas,
                           BoxVisual,
                           _tesselate_histogram)
from ._panzoom import PanZoomGrid
from ..utils._types import _as_array


#------------------------------------------------------------------------------
# CCG visual
#------------------------------------------------------------------------------

class CorrelogramVisual(BaseSpikeVisual):
    """Display a grid of auto- and cross-correlograms."""

    _shader_name = 'correlograms'
    _gl_draw_mode = 'triangle_strip'

    def __init__(self, **kwargs):
        super(CorrelogramVisual, self).__init__(**kwargs)
        self._correlograms = None
        self._cluster_ids = None
        self.n_samples = None

    # Data properties
    # -------------------------------------------------------------------------

    @property
    def correlograms(self):
        """Displayed correlograms.

        This is a `(n_clusters, n_clusters, n_bins)` array.

        """
        return self._correlograms

    @correlograms.setter
    def correlograms(self, value):
        value = _as_array(value)
        # WARNING: need to set cluster_ids first
        assert value.ndim == 3
        if self._cluster_ids is None:
            self._cluster_ids = np.arange(value.shape[0])
        assert value.shape[:2] == (self.n_clusters, self.n_clusters)
        self.n_samples = value.shape[2]
        self._correlograms = value
        self._empty = self.n_clusters == 0 or self.n_samples == 0
        self.set_to_bake('correlograms', 'color')

    @property
    def cluster_ids(self):
        """Displayed cluster ids."""
        return self._cluster_ids

    @cluster_ids.setter
    def cluster_ids(self, value):
        self._cluster_ids = np.asarray(value, dtype=np.int32)

    @property
    def n_boxes(self):
        """Number of boxes in the grid view."""
        return self.n_clusters * self.n_clusters

    # Data baking
    # -------------------------------------------------------------------------

    def _bake_correlograms(self):
        n_points = self.n_boxes * (5 * self.n_samples + 1)

        # index increases from top to bottom, left to right
        # same as matrix indices (i, j) starting at 0
        positions = []
        boxes = []

        for i in range(self.n_clusters):
            for j in range(self.n_clusters):
                index = self.n_clusters * i + j

                hist = self._correlograms[i, j, :]
                pos = _tesselate_histogram(hist)
                n_points_hist = pos.shape[0]

                positions.append(pos)
                boxes.append(index * np.ones(n_points_hist, dtype=np.float32))

        positions = np.vstack(positions).astype(np.float32)
        boxes = np.hstack(boxes)

        assert positions.shape == (n_points, 2)
        assert boxes.shape == (n_points,)

        self.program['a_position'] = positions.copy()
        self.program['a_box'] = boxes
        self.program['n_rows'] = self.n_clusters


class CorrelogramView(BaseSpikeCanvas):
    """A VisPy canvas displaying correlograms."""

    _visual_class = CorrelogramVisual

    def _create_visuals(self):
        super(CorrelogramView, self)._create_visuals()
        self.boxes = BoxVisual()

    def _create_pan_zoom(self):
        self._pz = PanZoomGrid()
        self._pz.add(self.visual.program)
        self._pz.attach(self)
        self._pz.aspect = None
        self._pz.zmin = 1.
        self._pz.xmin = -1.
        self._pz.xmax = +1.
        self._pz.ymin = -1.
        self._pz.ymax = +1.
        self._pz.zoom_to_pointer = False

    @property
    def cluster_ids(self):
        """Displayed cluster ids."""
        return self.visual.cluster_ids

    @cluster_ids.setter
    def cluster_ids(self, value):
        self.visual.cluster_ids = value
        self.boxes.n_rows = self.visual.n_clusters
        if self.visual.n_clusters >= 1:
            self._pz.n_rows = self.visual.n_clusters

    def on_draw(self, event):
        """Draw the correlograms visual."""
        gloo.clear()
        self.visual.draw()
        self.boxes.draw()


#------------------------------------------------------------------------------
# CCG plotting
#------------------------------------------------------------------------------

def plot_ccg(ccg, baseline=None, bin=1., color=None, ax=None):
    """Plot a CCG with matplotlib and return an Axes instance."""
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.subplot(111)
    assert ccg.ndim == 1
    n = ccg.shape[0]
    assert n % 2 == 1
    bin = float(bin)
    x_min = -n // 2 * bin - bin / 2
    x_max = (n // 2 - 1) * bin + bin / 2
    width = bin * 1.05
    left = np.linspace(x_min, x_max, n)
    ax.bar(left, ccg, facecolor=color, width=width, linewidth=0)
    if baseline is not None:
        ax.axhline(baseline, color='k', linewidth=2, linestyle='-')
    ax.axvline(color='k', linewidth=2, linestyle='--')

    ax.set_xlim(x_min, x_max + bin / 2)
    ax.set_ylim(0)

    # Only keep the bottom and left ticks.
    _bottom_left_frame(ax)

    return ax
