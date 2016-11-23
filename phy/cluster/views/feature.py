# -*- coding: utf-8 -*-

"""Feature view."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import inspect
from itertools import product
import logging
import re

import numpy as np

from phy.utils import Bunch
from phy.utils._color import _colormap
from .base import ManualClusteringView

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Feature view
# -----------------------------------------------------------------------------

def _extend(channels, n=None):
    channels = list(channels)
    if n is None:
        return channels
    if not len(channels):  # pragma: no cover
        channels = [0]
    if len(channels) < n:
        channels.extend([channels[-1]] * (n - len(channels)))
    channels = channels[:n]
    assert len(channels) == n
    return channels


def _dimensions_matrix(channels, n_cols=None, top_left_attribute=None):
    """
    time,x0 y0,x0   x1,x0   y1,x0
    x0,y0   time,y0 x1,y0   y1,y0
    x0,x1   y0,x1   time,x1 y1,x1
    x0,y1   y0,y1   x1,y1   time,y1
    """
    # Generate the dimensions matrix from the docstring.
    ds = inspect.getdoc(_dimensions_matrix).strip()
    x, y = channels[:2]

    def _get_dim(d):
        if d == 'time':
            return d
        assert re.match(r'[xy][01]', d)
        c = x if d[0] == 'x' else y
        f = int(d[1])
        return c, f

    dims = [[_.split(',') for _ in re.split(r' +', line.strip())]
            for line in ds.splitlines()]
    x_dim = {(i, j): _get_dim(dims[i][j][0])
             for i, j in product(range(4), range(4))}
    y_dim = {(i, j): _get_dim(dims[i][j][1])
             for i, j in product(range(4), range(4))}
    return x_dim, y_dim


class FeatureView(ManualClusteringView):
    _default_marker_size = 3.

    default_shortcuts = {
        'increase': 'ctrl++',
        'decrease': 'ctrl+-',
        'toggle_automatic_channel_selection': 'c',
    }

    def __init__(self,
                 features=None,
                 background_features=None,
                 spike_times=None,
                 n_channels=None,
                 n_features_per_channel=None,
                 best_channels=None,
                 **kwargs):
        """
        features is a function :
            `cluster_ids: Bunch(spike_ids,
                                features,
                                masks,
                                spike_clusters,
                                spike_times)`
        background_features is a Bunch(...) like above.

        """
        self._scaling = 1.

        self.best_channels = best_channels or (lambda clusters=None: [])

        assert features
        self.features = features

        # This is a tuple (spikes, features, masks).
        self.background_features = background_features

        self.n_features_per_channel = n_features_per_channel
        assert n_channels > 0
        self.n_channels = n_channels

        # Spike times.
        self.n_spikes = spike_times.shape[0]
        assert spike_times.shape == (self.n_spikes,)
        assert self.n_spikes >= 0
        self.spike_times = spike_times
        self.duration = spike_times.max()

        self.n_cols = 4
        self.shape = (self.n_cols, self.n_cols)

        # Initialize the view.
        super(FeatureView, self).__init__(layout='grid',
                                          shape=self.shape,
                                          enable_lasso=True,
                                          **kwargs)

        # If this is True, the channels won't be automatically chosen
        # when new clusters are selected.
        self.fixed_channels = False

        # Channels to show.
        self.channels = None

        # Attributes: extra features. This is a dictionary
        # {name: array}
        # where each array is a `(n_spikes,)` array.
        self.attributes = {}
        self.top_left_attribute = None

    # Internal methods
    # -------------------------------------------------------------------------

    def _get_feature(self, dim, spike_ids, f):
        if dim == 'time':
            return -1. + (2. / self.duration) * self.spike_times[spike_ids]
        elif dim in self.attributes:
            # Extra features.
            values = self.attributes[dim]
            values = values[spike_ids]
            return values
        else:
            assert len(dim) == 2
            ch, fet = dim
            assert fet < f.shape[2]
            return f[:, ch, fet] * self._scaling

    def _plot_features(self, i, j, x_dim, y_dim, x, y,
                       masks=None, clu_idx=None):
        """Plot the features in a subplot."""
        assert x.shape == y.shape
        n_spikes = x.shape[0]

        if clu_idx is not None:
            color = tuple(_colormap(clu_idx)) + (.5,)
        else:
            color = (1., 1., 1., .5)
        assert len(color) == 4

        # Find the masks for the given subplot channel.
        if isinstance(x_dim[i, j], tuple):
            ch, fet = x_dim[i, j]
            # NOTE: we add the cluster relative index for the computation
            # of the depth on the GPU.
            m = masks[:, ch] * .999 + (clu_idx or 0)
        else:
            m = np.ones(n_spikes) * .999 + (clu_idx or 0)

        # Marker size, smaller for background features.
        size = self._default_marker_size if clu_idx is not None else 1.

        self[i, j].scatter(x=x, y=y,
                           color=color,
                           masks=m,
                           size=size,
                           data_bounds=None,
                           uniform=True,
                           )

    def _plot_labels(self, x_dim, y_dim):
        """Plot feature labels along left and bottom edge of subplots"""

        # iterate simultaneously over kth row in left column and
        # kth column in bottom row:
        br = self.n_cols - 1  # bottom row
        for k in range(0, self.n_cols):
            label = str(y_dim[k, k])
            # left edge of left column of subplots:
            self[k, 0].text(pos=[-1., 0.],
                            text=label,
                            anchor=[-1.03, 0.],
                            data_bounds=None,
                            )
            # bottom edge of bottom row of subplots:
            self[br, k].text(pos=[0., -1.],
                             text=label,
                             anchor=[0., -1.04],
                             data_bounds=None,
                             )

    def _get_channel_dims(self, cluster_ids):
        """Select the channels to show by default."""
        n = 2
        channels = self.best_channels(cluster_ids)
        channels = (channels if channels is not None
                    else list(range(self.n_channels)))
        channels = _extend(channels, n)
        assert len(channels) == n
        return channels

    # Public methods
    # -------------------------------------------------------------------------

    def add_attribute(self, name, values, top_left=True):
        """Add an attribute (aka extra feature).

        The values should be a 1D array with `n_spikes` elements.

        NOTE: the values should be normalized by the caller.

        """
        assert values.shape == (self.n_spikes,)
        self.attributes[name] = values
        # Register the attribute to use in the top-left subplot.
        if top_left:
            self.top_left_attribute = name

    def clear_channels(self):
        """Reset the dimensions."""
        self.channels = None
        self.on_select()

    def on_select(self, cluster_ids=None):
        super(FeatureView, self).on_select(cluster_ids)
        cluster_ids = self.cluster_ids
        n_clusters = len(cluster_ids)
        if n_clusters == 0:
            return

        # Get the background features.
        data_bg = self.background_features
        if data_bg is not None:
            spike_ids_bg = data_bg.spike_ids
            features_bg = data_bg.data
            masks_bg = data_bg.masks
        # Select the dimensions.
        # Choose the channels automatically unless fixed_channels is set.
        if (not self.fixed_channels or self.channels is None):
            self.channels = self._get_channel_dims(cluster_ids)
        tla = self.top_left_attribute
        assert self.channels
        x_dim, y_dim = _dimensions_matrix(self.channels,
                                          n_cols=self.n_cols,
                                          top_left_attribute=tla)

        # Set the status message.
        ch = ', '.join(map(str, self.channels))
        self.set_status('Channels: {}'.format(ch))

        # Set a non-time attribute as y coordinate in the top-left subplot.
        attrs = sorted(self.attributes)
        # attrs.remove('time')
        if attrs:
            y_dim[0, 0] = attrs[0]

        # Plot all features.
        with self.building():
            self._plot_labels(x_dim, y_dim)

            # Cluster features.
            for clu_idx, cluster_id in enumerate(cluster_ids):

                # Get the spikes, features, masks.
                # TODO: this returns a dict {(i, j): array}
                d = self.features(cluster_id)

                f = d.data
                ns, nc = f.shape[:2]
                masks = d.get('masks', np.ones(f.shape[:2]))
                spike_ids = d.get('spike_ids', np.arange(ns))

                for i in range(self.n_cols):
                    for j in range(self.n_cols):
                        # Skip lower-diagonal subplots.
                        if i > j:
                            continue

                        if data_bg is not None:
                            # Retrieve the x and y values for the background
                            # spikes.
                            x_bg = self._get_feature(x_dim[i, j], spike_ids_bg,
                                                     features_bg)
                            y_bg = self._get_feature(y_dim[i, j], spike_ids_bg,
                                                     features_bg)

                            # Background features.
                            self._plot_features(i, j, x_dim, y_dim, x_bg, y_bg,
                                                masks=masks_bg,
                                                )

                        # Retrieve the x and y values for the subplot.
                        x = self._get_feature(x_dim[i, j], spike_ids, f)
                        y = self._get_feature(y_dim[i, j], spike_ids, f)

                        self._plot_features(i, j, x_dim, y_dim,
                                            x, y,
                                            masks=masks,
                                            clu_idx=clu_idx,
                                            )

                        # Add axes.
                        if clu_idx == 0:
                            self[i, j].lines(pos=[[-1., 0., +1., 0.],
                                                  [0., -1., 0., +1.]],
                                             color=(.25, .25, .25, .5))

            # Add the boxes.
            self.grid.add_boxes(self, self.shape)

    def attach(self, gui):
        """Attach the view to the GUI."""
        super(FeatureView, self).attach(gui)
        self.actions.add(self.increase)
        self.actions.add(self.decrease)
        self.actions.add(self.clear_channels)
        self.actions.add(self.toggle_automatic_channel_selection)

        gui.connect_(self.on_channel_click)
        gui.connect_(self.on_request_split)

    @property
    def state(self):
        return Bunch(scaling=self.scaling)

    def on_channel_click(self, channel_idx=None, key=None, button=None):
        """Respond to the click on a channel."""
        channels = self.channels
        if channels is None:
            return
        assert len(channels) == 2
        assert 0 <= channel_idx < self.n_channels
        # Get the axis from the pressed button (1, 2, etc.)
        # axis = 'x' if button == 1 else 'y'
        channels[0 if button == 1 else 1] = channel_idx
        self.fixed_channels = True
        self.on_select()

    def on_request_split(self):
        """Return the spikes enclosed by the lasso."""
        if self.lasso.count < 3:  # pragma: no cover
            return []
        tla = self.top_left_attribute
        assert self.channels
        x_dim, y_dim = _dimensions_matrix(self.channels,
                                          n_cols=self.n_cols,
                                          top_left_attribute=tla)
        data = self.features(self.cluster_ids, load_all=True)

        # Concatenate the points from all selected clusters.
        assert isinstance(data, list)
        pos = []
        for d in data:
            spike_ids = d.spike_ids
            f = d.data
            i, j = self.lasso.box

            x = self._get_feature(x_dim[i, j], spike_ids, f)
            y = self._get_feature(y_dim[i, j], spike_ids, f)
            pos.append(np.c_[x, y].astype(np.float64))
        pos = np.vstack(pos)

        ind = self.lasso.in_polygon(pos)
        self.lasso.clear()
        return spike_ids[ind]

    def toggle_automatic_channel_selection(self):
        """Toggle the automatic selection of channels when the cluster
        selection changes."""
        self.fixed_channels = not self.fixed_channels

    def increase(self):
        """Increase the scaling of the features."""
        self.scaling *= 1.2
        self.on_select()

    def decrease(self):
        """Decrease the scaling of the features."""
        self.scaling /= 1.2
        self.on_select()

    # Feature scaling
    # -------------------------------------------------------------------------

    @property
    def scaling(self):
        return self._scaling

    @scaling.setter
    def scaling(self, value):
        self._scaling = value
