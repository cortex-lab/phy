# -*- coding: utf-8 -*-

"""Color routines."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import colorcet as cc
import logging

from phylib.utils import Bunch
from phylib.io.array import _index_of

import numpy as np
from numpy.random import uniform
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Random colors
#------------------------------------------------------------------------------

def _random_color(h_range=(0., 1.), s_range=(.5, 1.), v_range=(.5, 1.)):
    """Generate a random RGB color."""
    h, s, v = uniform(*h_range), uniform(*s_range), uniform(*v_range)
    r, g, b = hsv_to_rgb(np.array([[[h, s, v]]])).flat
    return r, g, b


def _is_bright(rgb):
    """Return whether a RGB color is bright or not.
    see https://stackoverflow.com/a/3943023/1595060
    """
    L = 0
    for c, coeff in zip(rgb, (0.2126, 0.7152, 0.0722)):
        if c <= 0.03928:
            c = c / 12.92
        else:
            c = ((c + 0.055) / 1.055) ** 2.4
        L += c * coeff
    if (L + 0.05) / (0.0 + 0.05) > (1.0 + 0.05) / (L + 0.05):
        return True


def _random_bright_color():
    """Generate a random bright color."""
    rgb = _random_color()
    while not _is_bright(rgb):
        rgb = _random_color()
    return rgb


def _hex_to_triplet(h):
    """Convert an hexadecimal color to a triplet of int8 integers."""
    if h.startswith('#'):
        h = h[1:]
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _override_hsv(rgb, h=None, s=None, v=None):
    h_, s_, v_ = rgb_to_hsv(np.array([[rgb]])).flat
    h = h if h is not None else h_
    s = s if s is not None else s_
    v = v if v is not None else v_
    r, g, b = hsv_to_rgb(np.array([[[h, s, v]]])).flat
    return r, g, b


#------------------------------------------------------------------------------
# Colormap utilities
#------------------------------------------------------------------------------

def _selected_cluster_idx(selected_clusters, cluster_ids):
    selected_clusters = np.asarray(selected_clusters, dtype=np.int32)
    cluster_ids = np.asarray(cluster_ids, dtype=np.int32)
    kept = np.isin(selected_clusters, cluster_ids)
    clu_idx = _index_of(selected_clusters[kept], cluster_ids)
    cmap_idx = np.arange(len(selected_clusters))[kept]
    return clu_idx, cmap_idx


def _continuous_colormap(colormap, values, vmin=None, vmax=None):
    """Convert values into colors given a specified continuous colormap."""
    assert values is not None
    assert colormap.shape[1] == 3
    n = colormap.shape[0]
    vmin = vmin if vmin is not None else values.min()
    vmax = vmax if vmax is not None else values.max()
    assert vmin is not None
    assert vmax is not None
    denom = vmax - vmin
    denom = denom if denom != 0 else 1
    # NOTE: clipping is necessary when a view using color selector (like the raster view)
    # is updated right after a clustering update, but before the vmax had a chance to
    # be updated.
    i = np.clip(np.round((n - 1) * (values - vmin) / denom).astype(np.int32), 0, n - 1)
    return colormap[i, :]


def _categorical_colormap(colormap, values, vmin=None, vmax=None, categorize=None):
    """Convert values into colors given a specified categorical colormap."""
    assert np.issubdtype(values.dtype, np.integer)
    assert colormap.shape[1] == 3
    n = colormap.shape[0]
    if categorize is True or (categorize is None and vmin is None and vmax is None):
        # Find unique values and keep the order.
        _, idx = np.unique(values, return_index=True)
        lookup = values[np.sort(idx)]
        x = _index_of(values, lookup)
    else:
        x = values
    return colormap[x % n, :]


#------------------------------------------------------------------------------
# Colormaps
#------------------------------------------------------------------------------

# Default color map for the selected clusters.
# see https://colorcet.pyviz.org/user_guide/Categorical.html
def _make_default_colormap():
    """Return the default colormap, with custom first colors."""
    colormap = np.array(cc.glasbey_bw_minc_20_minl_30)
    # Reorder first colors.
    colormap[[0, 1, 2, 3, 4, 5]] = colormap[[3, 0, 4, 5, 2, 1]]
    # Replace first two colors.
    colormap[0] = [0.03137, 0.5725, 0.9882]
    colormap[1] = [1.0000, 0.0078, 0.0078]
    return colormap


def _make_cluster_group_colormap():
    """Return cluster group colormap."""
    return np.array([
        [0.4, 0.4, 0.4],  # noise
        [0.5, 0.5, 0.5],  # mua
        [0.5254, 0.8196, 0.42745],  # good
        [0.75, 0.75, 0.75],  # '' (None = '' = unsorted)
    ])


"""Built-in colormaps."""
colormaps = Bunch(
    blank=np.array([[.75, .75, .75]]),
    default=_make_default_colormap(),
    cluster_group=_make_cluster_group_colormap(),
    categorical=np.array(cc.glasbey_bw_minc_20_minl_30),
    rainbow=np.array(cc.rainbow_bgyr_35_85_c73),
    linear=np.array(cc.linear_wyor_100_45_c55),
    diverging=np.array(cc.diverging_linear_bjy_30_90_c45),
)


def selected_cluster_color(i, alpha=1.):
    """Return the color, as a 4-tuple, of the i-th selected cluster."""
    return add_alpha(tuple(colormaps.default[i % len(colormaps.default)]), alpha=alpha)


def spike_colors(spike_clusters, cluster_ids):
    """Return the colors of spikes according to the index of their cluster within `cluster_ids`.

    Parameters
    ----------

    spike_clusters : array-like
        The spike-cluster assignments.
    cluster_ids : array-like
        The set of unique selected cluster ids appearing in spike_clusters, in a given order

    Returns
    -------

    spike_colors : array-like
        For each spike, the RGBA color (in [0,1]) depending on the index of the cluster within
        `cluster_ids`.

    """
    spike_clusters_idx = _index_of(spike_clusters, cluster_ids)
    return add_alpha(colormaps.default[np.mod(spike_clusters_idx, colormaps.default.shape[0])])


def _add_selected_clusters_colors(selected_clusters, cluster_ids, cluster_colors=None):
    """Take an array with colors of clusters as input, and add colors of selected clusters."""
    # clu_idx contains the index of the selected clusters within cluster_ids
    # cmap_idx contains 0, 1, 2... as the colormap index, but without the selected clusters
    # that are missing in cluster_ids.
    clu_idx, cmap_idx = _selected_cluster_idx(selected_clusters, cluster_ids)
    colormap = _categorical_colormap(colormaps.default, cmap_idx, categorize=False)
    # Inject those colors in cluster_colors.
    cluster_colors[clu_idx] = add_alpha(colormap, 1)
    return cluster_colors


#------------------------------------------------------------------------------
# Cluster color selector
#------------------------------------------------------------------------------

def add_alpha(c, alpha=1.):
    """Add an alpha channel to an RGB color.

    Parameters
    ----------

    c : array-like (2D, shape[1] == 3) or 3-tuple
    alpha : float

    """
    if isinstance(c, (tuple,)):
        if len(c) == 4:
            c = c[:3]
        return c + (alpha,)
    elif isinstance(c, np.ndarray):
        if c.shape[-1] == 4:
            c = c[..., :3]
        assert c.shape[-1] == 3
        out = np.concatenate([c, alpha * np.ones((c.shape[:-1] + (1,)))], axis=-1)
        assert out.ndim == c.ndim
        assert out.shape[-1] == c.shape[-1] + 1
        return out
    raise ValueError("Unknown value given in add_alpha().")  # pragma: no cover


def _categorize(values):
    """Categorize a list of values by replacing strings and None values by integers."""
    if any(isinstance(v, str) for v in values):
        # HACK: replace None by empty string to avoid error when sorting the unique values.
        values = [str(v).lower() if v is not None else '' for v in values]
        uv = sorted(set(values))
        values = [uv.index(v) for v in values]
    return values


class ClusterColorSelector(object):
    """Assign a color to clusters depending on cluster labels or metrics."""
    _colormap = colormaps.categorical
    _categorical = True
    _logarithmic = False

    def __init__(
            self, fun=None, colormap=None, categorical=None, logarithmic=None, cluster_ids=None):
        self.cluster_ids = cluster_ids if cluster_ids is not None else ()
        self._fun = fun
        self.set_color_mapping(
            fun=fun, colormap=colormap, categorical=categorical, logarithmic=logarithmic)

    def set_color_mapping(
            self, fun=None, colormap=None, categorical=None, logarithmic=None):
        """Set the field used to choose the cluster colors, and the associated colormap.

        Parameters
        ----------

        fun : function
            Function cluster_id => value
        colormap : array-like
            A `(N, 3)` array with the colormaps colors
        categorical : boolean
            Whether the colormap is categorical (one value = one color) or continuous (values
            are continuously mapped from their initial interval to the colors).
        logarithmic : boolean
            Whether to use a logarithmic transform for the mapping.

        """
        self._fun = self._fun or fun
        if isinstance(colormap, str):
            colormap = colormaps[colormap]
        self._colormap = colormap if colormap is not None else self._colormap
        self._categorical = categorical if categorical is not None else self._categorical
        self._logarithmic = logarithmic if logarithmic is not None else self._logarithmic
        # Recompute the value range.
        self.set_cluster_ids(self.cluster_ids)

    def set_cluster_ids(self, cluster_ids):
        """Precompute the value range for all clusters."""
        self.cluster_ids = cluster_ids
        values = self.get_values(self.cluster_ids)
        if values is not None and len(values):
            self.vmin, self.vmax = values.min(), values.max()
        else:  # pragma: no cover
            self.vmin, self.vmax = 0, 1

    def map(self, values):
        """Convert values to colors using the selected colormap.

        Parameters
        ----------

        values : array-like (1D)

        Returns
        -------

        colors : array-like (2D, shape[1] == 3)

        """
        if self._logarithmic:
            assert np.all(values > 0)
            values = np.log(values)
            vmin, vmax = np.log(self.vmin), np.log(self.vmax)
        else:
            vmin, vmax = self.vmin, self.vmax
        assert values is not None
        # Use categorical or continuous colormap depending on the categorical option.
        f = (_categorical_colormap
             if self._categorical and np.issubdtype(values.dtype, np.integer)
             else _continuous_colormap)
        return f(self._colormap, values, vmin=vmin, vmax=vmax)

    def _get_cluster_value(self, cluster_id):
        """Return the field value for a given cluster."""
        return self._fun(cluster_id) if hasattr(self._fun, '__call__') else self._fun or 0

    def get(self, cluster_id, alpha=None):
        """Return the RGBA color of a single cluster."""
        assert self.cluster_ids is not None
        assert self._colormap is not None
        val = [self._get_cluster_value(cluster_id)]
        if self._categorical:
            val = _categorize(val)
        col = tuple(self.map(np.array(val))[0].tolist())
        return add_alpha(col, alpha=alpha)

    def get_values(self, cluster_ids):
        """Get the values of clusters for the selected color field.."""
        values = [self._get_cluster_value(cluster_id) for cluster_id in cluster_ids]
        if self._categorical:
            values = _categorize(values)
        return np.array(values)

    def get_colors(self, cluster_ids, alpha=1.):
        """Return the RGBA colors of some clusters."""
        values = self.get_values(cluster_ids)
        assert values is not None
        assert len(values) == len(cluster_ids)
        return add_alpha(self.map(values), alpha=alpha)
