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
                           AxisVisual,
                           _tesselate_histogram,
                           _wrap_vispy)
from ._panzoom import PanZoomGrid
from ..utils._types import _as_array, _as_list
from ..utils._color import _selected_clusters_colors


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
        self.n_bins = None

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
        self.n_bins = value.shape[2]
        self._correlograms = value
        self._empty = self.n_clusters == 0 or self.n_bins == 0
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
        n_points = self.n_boxes * (5 * self.n_bins + 1)

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
    _lines = []

    def _create_visuals(self):
        super(CorrelogramView, self)._create_visuals()
        self.boxes = BoxVisual()
        self.axes = AxisVisual()

    def _create_pan_zoom(self):
        self._pz = PanZoomGrid()
        self._pz.add(self.visual.program)
        self._pz.add(self.axes.program)
        self._pz.attach(self)
        self._pz.aspect = None
        self._pz.zmin = 1.
        self._pz.xmin = -1.
        self._pz.xmax = +1.
        self._pz.ymin = -1.
        self._pz.ymax = +1.
        self._pz.zoom_to_pointer = False

    def set_data(self,
                 correlograms=None,
                 colors=None,
                 lines=None):

        if correlograms is not None:
            correlograms = np.asarray(correlograms)
        else:
            correlograms = self.visual.correlograms
        assert correlograms.ndim == 3
        n_clusters = len(correlograms)
        assert correlograms.shape[:2] == (n_clusters, n_clusters)

        if colors is None:
            colors = _selected_clusters_colors(n_clusters)

        self.cluster_ids = np.arange(n_clusters)
        self.visual.correlograms = correlograms
        self.visual.cluster_colors = colors

        if lines is not None:
            self.lines = lines

        self.update()

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
        self.axes.n_rows = self.visual.n_clusters
        if self._lines:
            self.lines = self.lines

    @property
    def correlograms(self):
        return self.visual.correlograms

    @correlograms.setter
    def correlograms(self, value):
        self.visual.correlograms = value
        # Update the lines which depend on the number of bins.
        self.lines = self.lines

    @property
    def lines(self):
        """List of x coordinates where to put vertical lines.

        This is unit of samples.

        """
        return self._lines

    @lines.setter
    def lines(self, value):
        self._lines = _as_list(value)
        c = 2. / (float(max(1, self.visual.n_bins or 0)))
        self.axes.xs = np.array(self._lines) * c
        self.axes.color = (.5, .5, .5, 1.)

    @property
    def lines_color(self):
        return self.axes.color

    @lines_color.setter
    def lines_color(self, value):
        self.axes.color = value

    def on_draw(self, event):
        """Draw the correlograms visual."""
        gloo.clear()
        self.visual.draw()
        self.boxes.draw()
        if self._lines:
            self.axes.draw()


#------------------------------------------------------------------------------
# CCG plotting
#------------------------------------------------------------------------------

@_wrap_vispy
def plot_correlograms(correlograms, **kwargs):
    """Plot an array of correlograms.

    Parameters
    ----------

    correlograms : array
        A `(n_clusters, n_clusters, n_bins)` array.
    lines :  ndarray
        Array of x coordinates where to put vertical lines (in number of
        samples).
    colors : array-like (optional)
        A list of colors as RGB tuples.

    """
    c = CorrelogramView(keys='interactive')
    c.set_data(correlograms, **kwargs)
    return c


def _plot_ccg_mpl(ccg, baseline=None, bin=1., color=None, ax=None):
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
