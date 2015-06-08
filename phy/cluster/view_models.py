# -*- coding: utf-8 -*-

"""View model for clustered data."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op

import numpy as np

from ..utils.array import _unique, _spikes_in_clusters
from ..utils.selector import Selector
from ..utils._misc import _show_shortcuts
from ..utils import _as_list
from ..stats.ccg import correlograms, _symmetrize_correlograms
from ..plot.ccg import CorrelogramView
from ..plot.features import FeatureView
from ..plot.waveforms import WaveformView
from ..plot.traces import TraceView
from ..gui.base import BaseViewModel, HTMLViewModel


#------------------------------------------------------------------------------
# Misc
#------------------------------------------------------------------------------

# Default color map for the selected clusters.
_COLORMAP = np.array([[102, 194, 165],
                      [252, 141, 98],
                      [141, 160, 203],
                      [231, 138, 195],
                      [166, 216, 84],
                      [255, 217, 47],
                      [229, 196, 148],
                      ])


def _create_view(cls, backend=None, **kwargs):
    if backend in ('pyqt4', None):
        kwargs.update({'always_on_top': True})
    return cls(**kwargs)


def _selected_clusters_colors(n_clusters):
    if n_clusters > _COLORMAP.shape[0]:
        colors = np.tile(_COLORMAP, (1 + n_clusters // _COLORMAP.shape[0], 1))
    else:
        colors = _COLORMAP
    return colors[:n_clusters, ...] / 255.


def _oddify(x):
    return x if x % 2 == 1 else x + 1


#------------------------------------------------------------------------------
# Base view models
#------------------------------------------------------------------------------

class BaseClusterViewModel(BaseViewModel):
    """Interface between a view and a model."""
    _view_class = None

    def __init__(self, model=None,
                 store=None, wizard=None,
                 cluster_ids=None, **kwargs):
        assert store is not None
        self._store = store
        self._wizard = wizard

        super(BaseClusterViewModel, self).__init__(model=model,
                                                   **kwargs)

        self._cluster_ids = None
        if cluster_ids is not None:
            self.select(_as_list(cluster_ids))

    @property
    def store(self):
        return self._store

    @property
    def wizard(self):
        return self._wizard

    @property
    def cluster_ids(self):
        """Selected clusters."""
        return self._cluster_ids

    @property
    def n_clusters(self):
        """Number of selected clusters."""
        return len(self._cluster_ids)

    # Public methods
    #--------------------------------------------------------------------------

    def select(self, cluster_ids):
        """Select a set of clusters."""
        cluster_ids = _as_list(cluster_ids)
        self._cluster_ids = cluster_ids
        self.on_select(cluster_ids)

    # Callback methods
    #--------------------------------------------------------------------------

    def on_select(self, cluster_ids):
        """Update the view after a new selection has been made.

        Must be overriden."""

    def on_cluster(self, up):
        """Called when a clustering action occurs.

        May be overriden."""


class HTMLClusterViewModel(BaseClusterViewModel, HTMLViewModel):
    """HTML view model that displays per-cluster information."""

    # To be overriden
    #--------------------------------------------------------------------------

    _static_path = None
    _html_filename = None
    _html = ''

    def _format_dict(self, **kwargs):
        return {}

    # Callbacks
    #--------------------------------------------------------------------------

    def on_select(self, cluster_ids):
        self.update(cluster_ids=cluster_ids)

    def on_cluster(self, up):
        self.update(cluster_ids=self._cluster_ids, up=up)


class StatsViewModel(HTMLClusterViewModel):
    _static_path = op.join(op.dirname(op.realpath(__file__)), 'manual/static')

    def get_html(self, cluster_ids=None, up=None):
        stats = self.store.items['statistics']
        names = stats.fields
        if cluster_ids is None:
            return ''
        cluster = cluster_ids[0]
        html = ''
        for name in names:
            value = getattr(self.store, name)(cluster)
            if not isinstance(value, np.ndarray):
                html += '<p>{}: {}</p>\n'.format(name, value)
        return '<div class="stats">\n' + html + '</div>'


class VispyViewModel(BaseClusterViewModel):
    """Create a VisPy view from a model.

    This object uses an internal `Selector` instance to manage spike and
    cluster selection.

    """
    _imported_params = ('n_spikes_max', 'excerpt_size')
    keyboard_shortcuts = {}
    scale_factor = 1.

    def __init__(self, **kwargs):
        super(VispyViewModel, self).__init__(**kwargs)

        # Call on_close() when the view is closed.
        @self._view.connect
        def on_close(e):
            self.on_close()

    def _create_view(self, **kwargs):
        n_spikes_max = kwargs.get('n_spikes_max', None)
        excerpt_size = kwargs.get('excerpt_size', None)
        backend = kwargs.get('backend', None)
        position = kwargs.get('position', None)
        size = kwargs.get('size', None)

        # Create the spike/cluster selector.
        self._selector = Selector(self._model.spike_clusters,
                                  n_spikes_max=n_spikes_max,
                                  excerpt_size=excerpt_size,
                                  )

        # Create the VisPy canvas.
        view = _create_view(self._view_class,
                            backend=backend,
                            position=position or (200, 200),
                            size=size or (600, 600),
                            )
        view.connect(self.on_key_press)
        return view

    @property
    def selector(self):
        return self._selector

    @property
    def cluster_ids(self):
        """Selected clusters."""
        return self._selector.selected_clusters

    @property
    def spike_ids(self):
        """Selected spikes."""
        return self._selector.selected_spikes

    @property
    def n_spikes(self):
        """Number of selected spikes."""
        return self._selector.n_spikes

    def update_spike_clusters(self, spikes=None, spike_clusters=None):
        """Update the spike clusters and cluster colors."""
        if spikes is None:
            spikes = self.spike_ids
        if spike_clusters is None:
            spike_clusters = self.model.spike_clusters[spikes]
        n_clusters = len(_unique(spike_clusters))
        visual = self._view.visual
        # This updates the list of unique clusters in the view.
        visual.spike_clusters = spike_clusters
        visual.cluster_colors = _selected_clusters_colors(n_clusters)

    def select(self, cluster_ids):
        """Select a set of clusters."""
        self._selector.selected_clusters = cluster_ids
        self.on_select(cluster_ids)

    def on_select(self, cluster_ids):
        """Update the view after a new selection has been made.

        Must be overriden."""
        self.update_spike_clusters()
        self._view.update()

    def on_close(self):
        """Clear the view when the model is closed."""
        self._view.visual.spike_clusters = []
        self._view.update()

    def on_key_press(self, event):
        if event.key == 'h' and 'control' not in event.modifiers:
            shortcuts = self._view.keyboard_shortcuts
            shortcuts.update(self.keyboard_shortcuts)
            _show_shortcuts(shortcuts, name=self.name)


#------------------------------------------------------------------------------
# Kwik view models
#------------------------------------------------------------------------------

class WaveformViewModel(VispyViewModel):
    _view_class = WaveformView
    _view_name = 'waveforms'
    _imported_params = ('scale_factor', 'box_scale', 'probe_scale',
                        'overlap', 'show_mean')

    def on_open(self):
        super(WaveformViewModel, self).on_open()
        # Waveforms.
        self.view.visual.channel_positions = self.model.probe.positions
        self.view.visual.channel_order = self.model.channel_order
        # Mean waveforms.
        self.view.mean.channel_positions = self.model.probe.positions
        self.view.mean.channel_order = self.model.channel_order
        if self.scale_factor is None:
            self.scale_factor = 1.

    def _load_waveforms(self):
        # NOTE: we load all spikes from the store.
        # The waveforms store item is responsible for making a subselection
        # of spikes both on disk and in the view.
        waveforms = self.store.load('waveforms',
                                    clusters=self.cluster_ids,
                                    )
        return waveforms

    def _load_mean_waveforms(self):
        mean_waveforms = self.store.load('mean_waveforms',
                                         clusters=self.cluster_ids,
                                         )
        mean_masks = self.store.load('mean_masks',
                                     clusters=self.cluster_ids,
                                     )
        return mean_waveforms, mean_masks

    def update_spike_clusters(self, spikes=None):
        super(WaveformViewModel, self).update_spike_clusters(spikes=spikes)
        self._view.mean.spike_clusters = np.sort(self.cluster_ids)
        self._view.mean.cluster_colors = self._view.visual.cluster_colors

    def on_select(self, clusters):
        # Get the spikes of the stored waveforms.
        n_clusters = len(clusters)
        waveforms = self._load_waveforms()
        spikes = self.store.items['waveforms'].spikes_in_clusters(clusters)
        n_spikes = len(spikes)
        _, self._n_samples, self._n_channels = waveforms.shape
        mean_waveforms, mean_masks = self._load_mean_waveforms()

        self.update_spike_clusters(spikes)

        # Cluster display order.
        self.view.visual.cluster_order = clusters
        self.view.mean.cluster_order = clusters

        # Waveforms.
        assert waveforms.shape[0] == n_spikes
        self.view.visual.waveforms = waveforms * self.scale_factor

        assert mean_waveforms.shape == (n_clusters,
                                        self._n_samples,
                                        self._n_channels)
        self.view.mean.waveforms = mean_waveforms * self.scale_factor

        # Masks.
        masks = self.store.load('masks', clusters=clusters, spikes=spikes)
        assert masks.shape == (n_spikes, self._n_channels)
        self.view.visual.masks = masks

        assert mean_masks.shape == (n_clusters, self._n_channels)
        self.view.mean.masks = mean_masks

        # Spikes.
        self.view.visual.spike_ids = spikes
        self.view.mean.spike_ids = np.arange(len(clusters))

        self.view.update()

    def on_close(self):
        self.view.visual.channel_positions = []
        self.view.mean.channel_positions = []
        super(WaveformViewModel, self).on_close()

    @property
    def box_scale(self):
        """Scale of the waveforms.

        This is a pair of scalars.

        """
        return self.view.box_scale

    @box_scale.setter
    def box_scale(self, value):
        self.view.box_scale = value

    @property
    def probe_scale(self):
        """Scale of the probe.

        This is a pair of scalars.

        """
        return self.view.probe_scale

    @probe_scale.setter
    def probe_scale(self, value):
        self.view.probe_scale = value

    @property
    def overlap(self):
        """Whether to overlap waveforms."""
        return self.view.overlap

    @overlap.setter
    def overlap(self, value):
        self.view.overlap = value

    @property
    def show_mean(self):
        """Whether to show mean waveforms."""
        return self.view.show_mean

    @show_mean.setter
    def show_mean(self, value):
        self.view.show_mean = value

    def exported_params(self, save_size_pos=True):
        params = super(WaveformViewModel, self).exported_params(save_size_pos)
        params.update({
            'scale_factor': self.scale_factor,
            'box_scale': self.view.box_scale,
            'probe_scale': self.view.probe_scale,
            'overlap': self.view.overlap,
            'show_mean': self.view.show_mean,
        })
        return params


class CorrelogramViewModel(VispyViewModel):
    _view_class = CorrelogramView
    _view_name = 'correlograms'
    binsize = 20
    winsize_bins = 41
    _imported_params = ('binsize', 'winsize_bins')

    def change_bins(self, bin=None, half_width=None):
        """Change the parameters of the correlograms.

        Parameters
        ----------
        bin : float (ms)
            Bin size.
        half_width : float (ms)
            Half window size.

        """
        sr = self.model.sample_rate

        bin = np.clip(bin * .001, .001, 1e6)
        self.binsize = int(sr * bin)

        half_width = np.clip(half_width * .001, .001, 1e6)
        self.winsize_bins = 2 * int(half_width / bin) + 1

        self.select(self.cluster_ids)

    def on_select(self, clusters):
        super(CorrelogramViewModel, self).on_select(clusters)
        spikes = self.spike_ids
        self.view.cluster_ids = clusters

        # Compute the correlograms.
        spike_samples = self.model.spike_samples[spikes]
        spike_clusters = self.view.visual.spike_clusters

        ccgs = correlograms(spike_samples,
                            spike_clusters,
                            cluster_order=clusters,
                            binsize=self.binsize,
                            # NOTE: this must be an odd number, for symmetry
                            winsize_bins=_oddify(self.winsize_bins),
                            )
        ccgs = _symmetrize_correlograms(ccgs)
        # Normalize the CCGs.
        if len(ccgs):
            ccgs = ccgs * (1. / max(1., ccgs.max()))
        self.view.visual.correlograms = ccgs

        # Take the cluster order into account.
        self.view.visual.cluster_order = clusters
        self.view.update()


class TraceViewModel(VispyViewModel):
    _view_class = TraceView
    _view_name = 'traces'
    _imported_params = ('scale_factor', 'channel_scale', 'interval_size')
    interval_size = .25

    def __init__(self, **kwargs):
        self._interval = None
        super(TraceViewModel, self).__init__(**kwargs)

    def _load_traces(self, interval):
        start, end = interval
        spikes = self.spike_ids

        # Load the traces.
        # debug("Loading traces...")
        # Using channel_order ensures that we get rid of the dead channels.
        # We also keep the channel order as specified by the PRM file.
        # WARNING: HDF5 does not support out-of-order indexing (...!!)
        traces = self.model.traces[start:end, :][:, self.model.channel_order]

        # Normalize and set the traces.
        traces_f = np.empty_like(traces, dtype=np.float32)
        traces_f[...] = traces * self.scale_factor
        # Detrend the traces.
        m = np.mean(traces_f[::10, :], axis=0)
        traces_f -= m
        self.view.visual.traces = traces_f

        # Keep the spikes in the interval.
        spike_samples = self.model.spike_samples[spikes]
        a, b = spike_samples.searchsorted(interval)
        spikes = spikes[a:b]
        self.view.visual.n_spikes = len(spikes)
        self.view.visual.spike_ids = spikes

        if len(spikes) == 0:
            return

        # We update the spike clusters according to the subselection of spikes.
        # We don't update the list of unique clusters, which only change
        # when selecting or clustering, not when changing the interval.
        # self.update_spike_clusters(spikes)
        self.view.visual.spike_clusters = self.model.spike_clusters[spikes]

        # Set the spike samples.
        spike_samples = self.model.spike_samples[spikes]
        # This is in unit of samples relative to the start of the interval.
        spike_samples = spike_samples - start
        self.view.visual.spike_samples = spike_samples
        self.view.visual.offset = start

        # Load the masks.
        # TODO: ensure model.masks is always 2D, even with 1 spike
        masks = np.atleast_2d(self._model.masks[spikes])
        self.view.visual.masks = masks

    @property
    def interval(self):
        """The interval of the view, in unit of sample."""
        return self._interval

    @interval.setter
    def interval(self, value):
        if not isinstance(value, tuple) or len(value) != 2:
            raise ValueError("The interval should be a (start, end) tuple.")
        # Restrict the interval to the boundaries of the traces.
        start, end = value
        start, end = int(start), int(end)
        n = self.model.traces.shape[0]
        if start < 0:
            end += (-start)
            start = 0
        elif end >= n:
            start -= (end - n)
            end = n
        start = np.clip(start, 0, end)
        end = np.clip(end, start, n)
        assert 0 <= start < end <= n
        self._interval = (start, end)
        self._load_traces((start, end))
        self.view.update()

    @property
    def channel_scale(self):
        """Vertical scale of the traces."""
        return self.view.channel_scale

    @channel_scale.setter
    def channel_scale(self, value):
        self.view.channel_scale = value

    def move(self, amount):
        """Move the current interval by a given amount (in samples)."""
        amount = int(amount)
        start, end = self.interval
        self.interval = start + amount, end + amount

    def move_right(self, fraction=.05):
        """Move the current interval to the right."""
        start, end = self.interval
        self.move(int(+(end - start) * fraction))

    def move_left(self, fraction=.05):
        """Move the current interval to the left."""
        start, end = self.interval
        self.move(int(-(end - start) * fraction))

    keyboard_shortcuts = {
        'scroll_left': 'ctrl+left',
        'scroll_right': 'ctrl+right',
        'fast_scroll_left': 'shift+left',
        'fast_scroll_right': 'shift+right',
    }

    def on_key_press(self, event):
        super(TraceViewModel, self).on_key_press(event)
        key = event.key
        if 'Control' in event.modifiers:
            if key == 'Left':
                self.move_left()
            elif key == 'Right':
                self.move_right()
        if 'Shift' in event.modifiers:
            if key == 'Left':
                self.move_left(1)
            elif key == 'Right':
                self.move_right(1)

    def on_open(self):
        super(TraceViewModel, self).on_open()
        self.view.visual.n_samples_per_spike = self.model.n_samples_waveforms
        self.view.visual.sample_rate = self.model.sample_rate
        if self.scale_factor is None:
            self.scale_factor = 1.
        if self.interval_size is None:
            self.interval_size = .25
        self.select([])

    def on_select(self, clusters):
        # Get the spikes in the selected clusters.
        spikes = self.spike_ids
        n_clusters = len(clusters)
        spike_clusters = self.model.spike_clusters[spikes]

        # Update the clusters of the trace view.
        visual = self._view.visual
        visual.spike_clusters = spike_clusters
        visual.cluster_ids = clusters
        visual.cluster_order = clusters
        visual.cluster_colors = _selected_clusters_colors(n_clusters)

        # Select the default interval.
        half_size = int(self.interval_size * self.model.sample_rate / 2.)
        if len(spikes) > 0:
            # Center the default interval around the first spike.
            sample = self._model.spike_samples[spikes[0]]
        else:
            sample = half_size
        # Load traces by setting the interval.
        visual._update_clusters_automatically = False
        self.interval = sample - half_size, sample + half_size

    def exported_params(self, save_size_pos=True):
        params = super(TraceViewModel, self).exported_params(save_size_pos)
        params.update({
            'scale_factor': self.scale_factor,
            'channel_scale': self.channel_scale,
        })
        return params


#------------------------------------------------------------------------------
# Feature view models
#------------------------------------------------------------------------------

class BaseFeatureViewModel(VispyViewModel):
    _view_class = FeatureView
    _view_name = 'base_features'
    _imported_params = ('scale_factor', 'n_spikes_max_bg', 'marker_size')
    n_spikes_max_bg = 10000

    def _rescale_features(self, features):
        # WARNING: convert features to a 3D array
        # (n_spikes, n_channels, n_features)
        # because that's what the FeatureView expects currently.
        n_fet = self.model.n_features_per_channel
        n_channels = len(self.model.channel_order)
        shape = (-1, n_channels, n_fet)
        features = features[:, :n_fet * n_channels].reshape(shape)
        # Scale factor.
        return features * self.scale_factor

    @property
    def lasso(self):
        return self.view.lasso

    def spikes_in_lasso(self):
        """Return the spike ids from the selected clusters within the lasso."""
        if not len(self.cluster_ids) or self.view.lasso.n_points <= 2:
            return
        clusters = self.cluster_ids
        features = self.store.load('features', clusters=clusters)
        features = self._rescale_features(features)
        box = self.view.lasso.box
        points = self.view.visual.project(features, box)
        in_lasso = self.view.lasso.in_lasso(points)
        spike_ids = _spikes_in_clusters(self.model.spike_clusters, clusters)
        return spike_ids[in_lasso]

    @property
    def marker_size(self):
        """Marker size."""
        return self.view.marker_size

    @marker_size.setter
    def marker_size(self, value):
        self.view.marker_size = value

    def _alternative_dimension(self, dim):
        if dim == 'time':
            return (0, 0)
        else:
            channel, fet = dim
            return (channel, (fet + 1) % self.view.background.n_features)

    def _matrix_from_dimensions(self, dimensions):
        n = len(dimensions)
        matrix = np.empty((n, n), dtype=object)
        for i in range(n):
            for j in range(n):
                dim_i, dim_j = dimensions[i], dimensions[j]
                if dim_i == dim_j:
                    dim_j = self._alternative_dimension(dim_i)
                matrix[i, j] = (dim_i, dim_j)
        return matrix

    @property
    def dimensions_matrix(self):
        """The matrix of displayed dimensions."""
        return self._view.dimensions_matrix

    @dimensions_matrix.setter
    def dimensions_matrix(self, value):
        self._view.dimensions_matrix = value

    def _set_dimensions_after_open(self):
        matrix = self._matrix_from_dimensions(['time'])
        self.view.update_dimensions_matrix(matrix)
        self.view.update()

    def _set_dimensions_after_select(self):
        # Update the dimensions.
        self.view.dimensions_matrix = self.view.dimensions_matrix
        self.view.update()

    def on_open(self):
        # Get background features.
        # TODO OPTIM: precompute this once for all and store in the cluster
        # store. But might be unnecessary.
        if self.n_spikes_max_bg is not None:
            k = max(1, self.model.n_spikes // self.n_spikes_max_bg)
        else:
            k = 1
        if self.model.features is not None:
            # Background features.
            features_bg = self.store.load('features',
                                          spikes=slice(None, None, k))
            self.view.background.features = self._rescale_features(features_bg)
            # Time dimension.
            spike_samples = self.model.spike_samples[::k]
            self.view.background.spike_samples = spike_samples
        # Default dimensions.
        self._set_dimensions_after_open()

    def on_select(self, clusters):
        super(BaseFeatureViewModel, self).on_select(clusters)
        spikes = self.spike_ids

        features = self.store.load('features',
                                   clusters=clusters,
                                   spikes=spikes)
        masks = self.store.load('masks',
                                clusters=clusters,
                                spikes=spikes)

        nc = len(self.model.channel_order)
        nf = self.model.n_features_per_channel
        features = features.reshape((len(spikes), nc, nf))
        self.view.visual.features = self._rescale_features(features)
        self.view.visual.masks = masks

        # Spikes.
        self.view.visual.spike_ids = spikes
        self.view.visual.spike_samples = self.model.spike_samples[spikes]

        # Cluster display order.
        self.view.visual.cluster_order = clusters

        # Default dimensions.
        self._set_dimensions_after_select()

    def exported_params(self, save_size_pos=True):
        params = super(BaseFeatureViewModel,
                       self).exported_params(save_size_pos)
        zoom = self._view._pz.zoom
        params.update({
            'scale_factor': zoom.mean() * self.scale_factor,
            'marker_size': self.marker_size,
        })
        return params


class MultiFeatureViewModel(BaseFeatureViewModel):
    _view_name = 'features_grid'

    def __init__(self, **kwargs):
        self._dimension_selector = None
        super(MultiFeatureViewModel, self).__init__(**kwargs)

    def set_dimension_selector(self, func):
        """Decorator for a function that selects the best projection.

        The decorated function must have the following signature:

        ```python
        @view_model.set_dimension_selector
        def choose(cluster_ids):
            # ...
            return channel_idxs  # a list with 3 relative channel indices
        ```

        """
        self._dimension_selector = func

    def default_dimension_selector(self, cluster_ids):
        """Return the channels with the largest mean features.

        The first cluster is used currently.

        """
        if cluster_ids is None or not len(cluster_ids):
            return np.arange(len(self.model.channels[:3]))
        n_fet = self.model.n_features_per_channel
        score = self.store.mean_features(cluster_ids[0])
        score = score.reshape((-1, n_fet)).mean(axis=1)
        # Take the best 3 channels.
        assert len(score) == len(self.model.channel_order)
        channels = np.argsort(score)[::-1][:3]
        return channels

    @property
    def default_dimensions(self):
        dimension_selector = (self._dimension_selector or
                              self.default_dimension_selector)
        channels = dimension_selector(self.cluster_ids)
        return ['time'] + [(ch, 0) for ch in channels]

    def _set_dimensions_after_open(self):
        matrix = self._matrix_from_dimensions(self.default_dimensions)
        self.view.update_dimensions_matrix(matrix)
        self.view.update()

    def _set_dimensions_after_select(self):
        matrix = self._matrix_from_dimensions(self.default_dimensions)
        self.view.dimensions_matrix = matrix
        self.view.update()

    keyboard_shortcuts = {
        'enlarge_subplot': 'ctrl+click',
    }


class SingleFeatureViewModel(BaseFeatureViewModel):
    _view_name = 'features'

    def _set_dimensions_after_select(self):
        self.view.dimensions_matrix = self.view.dimensions_matrix
        self.view.update()

    def set_dimension(self, dim):
        matrix = self._matrix_from_dimensions([dim])
        self.view.dimensions_matrix = matrix

    def set_dimensions(self, dim_x, dim_y):
        matrix = self.view.dimensions_matrix
        matrix[0, 0] = (dim_x, dim_y)
        self.view.dimensions_matrix = matrix
