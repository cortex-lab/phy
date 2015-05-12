# -*- coding: utf-8 -*-

"""Views for Kwik model."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ...utils.logging import debug
from ...utils.array import _unique, get_excerpts
from ...plot.ccg import CorrelogramView
from ...plot.features import FeatureView
from ...plot.waveforms import WaveformView
from ...plot.traces import TraceView
from ...stats.ccg import correlograms, _symmetrize_correlograms
from .selector import Selector
from ._utils import _concatenate_per_cluster_arrays


#------------------------------------------------------------------------------
# BaseViewModel for plot views and Kwik model
#------------------------------------------------------------------------------

def _create_view(cls, backend=None, **kwargs):
    if backend in ('pyqt4', None):
        kwargs.update({'always_on_top': True})
    return cls(**kwargs)


_COLORMAP = np.array([[102, 194, 165],
                      [252, 141, 98],
                      [141, 160, 203],
                      [231, 138, 195],
                      [166, 216, 84],
                      [255, 217, 47],
                      [229, 196, 148],
                      ])


def _selected_clusters_colors(n_clusters):
    if n_clusters > _COLORMAP.shape[0]:
        colors = np.tile(_COLORMAP, (1 + n_clusters // _COLORMAP.shape[0], 1))
    else:
        colors = _COLORMAP
    return colors[:n_clusters, ...] / 255.


class BaseViewModel(object):
    """Used to create views from a model."""
    _view_class = None
    _view_name = ''

    def __init__(self, model, store=None,
                 n_spikes_max=None, excerpt_size=None,
                 **kwargs):
        self._model = model
        self._store = store

        # Create the spike/cluster selector.
        self._selector = Selector(model.spike_clusters,
                                  n_spikes_max=n_spikes_max,
                                  excerpt_size=excerpt_size,
                                  )

        # Set all keyword arguments as attributes.
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Extract VisPy keyword arguments.
        vispy_kwargs_names = ('position', 'size',)
        vispy_kwargs = {name: kwargs[name] for name in vispy_kwargs_names
                        if name in kwargs}
        backend = kwargs.pop('backend', None)

        # Create the VisPy canvas.
        self._view = _create_view(self._view_class,
                                  backend=backend,
                                  **vispy_kwargs)

        # Bind VisPy event methods.
        for method in ('on_key_press', 'on_mouse_move'):
            if hasattr(self, method):
                self._view.connect(getattr(self, method))

    @property
    def model(self):
        return self._model

    @property
    def view_name(self):
        return self._view_name

    @property
    def store(self):
        return self._store

    @property
    def selector(self):
        return self._selector

    @property
    def view(self):
        return self._view

    @property
    def cluster_ids(self):
        return self._selector.selected_clusters

    @property
    def spike_ids(self):
        return self._selector.selected_spikes

    @property
    def n_clusters(self):
        return self._selector.n_clusters

    @property
    def n_spikes(self):
        return self._selector.n_spikes

    def _load_from_store_or_model(self,
                                  name,
                                  cluster_ids,
                                  spikes,
                                  ):
        if self._store is not None:
            return self._store.load(name,
                                    cluster_ids,
                                    spikes,
                                    )
        else:
            return getattr(self._model, name)[spikes]

    def _update_spike_clusters(self, spikes=None):
        """Update the spike clusters and cluster colors."""
        if spikes is None:
            spikes = self.spike_ids
        spike_clusters = self.model.spike_clusters[spikes]
        n_clusters = len(_unique(spike_clusters))
        visual = self._view.visual
        # This updates the list of unique clusters in the view.
        visual.spike_clusters = spike_clusters
        visual.cluster_colors = _selected_clusters_colors(n_clusters)

    def on_open(self):
        """May be overriden."""

    def on_select(self, cluster_ids):
        """Must be overriden."""
        self._selector.selected_clusters = cluster_ids
        self._update_spike_clusters()

    def on_close(self):
        self._view.visual.spike_clusters = []
        self._view.update()

    def show(self):
        self._view.show()


#------------------------------------------------------------------------------
# View models
#------------------------------------------------------------------------------

class WaveformViewModel(BaseViewModel):
    _view_class = WaveformView
    _view_name = 'waveforms'
    scale_factor = 1.

    def on_open(self):
        super(WaveformViewModel, self).on_open()
        self.view.visual.channel_positions = self.model.probe.positions
        self.view.visual.channel_order = self.model.channel_order

    def on_select(self, cluster_ids):

        # Get the spikes of the stored waveforms.
        debug("Loading waveforms...")
        if self._store is not None:
            # Subset the stored spikes for each cluster.
            k = len(cluster_ids)
            spc = {cluster: self._store.waveforms_spikes(cluster)[::k]
                   for cluster in cluster_ids}
            # Get all the spikes for the clusters.
            spikes = _concatenate_per_cluster_arrays(spc, spc)
            # Subset the spikes with the selector.
            self._selector.selected_spikes = spikes
            spikes = self.spike_ids
            # Load the waveforms for the subset spikes.
            waveforms = self._store.load('waveforms',
                                         cluster_ids,
                                         spikes,
                                         spikes_per_cluster=spc,
                                         )
        else:
            # If there's no store, just take the waveforms from the traces
            # (slower).
            self._selector.selected_clusters = cluster_ids
            spikes = self.spike_ids
            waveforms = self.model.waveforms[spikes]

        self._update_spike_clusters()
        assert waveforms.shape[0] == len(spikes)
        debug("Done!")

        # Cluster display order.
        self.view.visual.cluster_order = cluster_ids

        # Waveforms.
        waveforms *= self.scale_factor
        self.view.visual.waveforms = waveforms

        # Masks.
        masks = self._load_from_store_or_model('masks', cluster_ids, spikes)
        self.view.visual.masks = masks

        # Spikes.
        self.view.visual.spike_ids = spikes

    def on_close(self):
        self.view.visual.channel_positions = []
        super(WaveformViewModel, self).on_close()


class FeatureViewModel(BaseViewModel):
    _view_class = FeatureView
    _view_name = 'features'
    scale_factor = 1.
    _dimension_selector = None
    n_spikes_max_bg = 10000
    excerpt_size_bg = 1000

    def set_dimension_selector(self, func):
        """Decorator for a function that selects the best projection.

        The decorated function must have the following signature:

        ```python
        @view_model.set_dimension_selector
        def choose(cluster_ids, store=None):
            # ...
            return channel_idxs  # a list with 3 relative channel indices
        ```

        """
        self._dimension_selector = func

    def _rescale_features(self, features):
        # WARNING: convert features to a 3D array
        # (n_spikes, n_channels, n_features)
        # because that's what the FeatureView expects currently.
        n_fet = self.model.n_features_per_channel
        n_channels = len(self.model.channel_order)
        shape = (-1, n_channels, n_fet)
        features = features[:, :n_fet * n_channels].reshape(shape)
        # Scale factor.
        features *= self.scale_factor
        return features

    def on_open(self):
        # Get background features.
        # self._selector_bg = Selector(self.model.spike_clusters,
        #                              n_spikes_max=self.n_spikes_max_bg,
        #                              excerpt_size=self.excerpt_size_bg,
        #                              )
        # self._selector_bg.selected_spikes = np.arange(self.model.n_spikes)
        # spikes = self._selector_bg.selected_spikes
        n_excerpts = self.n_spikes_max_bg // self.excerpt_size_bg
        features_bg = get_excerpts(self.model.features,
                                   n_excerpts=n_excerpts,
                                   excerpt_size=self.excerpt_size_bg,
                                   )
        spike_samples = get_excerpts(self.model.spike_samples,
                                     n_excerpts=n_excerpts,
                                     excerpt_size=self.excerpt_size_bg,
                                     )
        self.view.background.features = self._rescale_features(features_bg)
        self.view.background.spike_samples = spike_samples

    def on_select(self, cluster_ids):
        super(FeatureViewModel, self).on_select(cluster_ids)
        spikes = self.spike_ids

        # Load features.
        features = self._load_from_store_or_model('features',
                                                  cluster_ids,
                                                  spikes)
        # Load masks.
        masks = self._load_from_store_or_model('masks',
                                               cluster_ids,
                                               spikes)

        self.view.visual.features = self._rescale_features(features)
        self.view.visual.masks = masks

        # Spikes.
        self.view.visual.spike_ids = spikes
        self.view.visual.spike_samples = self.model.spike_samples[spikes]

        # Cluster display order.
        self.view.visual.cluster_order = cluster_ids

        # Choose best projection.
        # TODO: refactor this, enable/disable
        if self.store and self._dimension_selector is not None:
            channels = self._dimension_selector(cluster_ids, store=self.store)
        else:
            channels = np.arange(len(self.model.channels[:3]))
        self.view.dimensions = ['time'] + [(ch, 0) for ch in channels]


class CorrelogramViewModel(BaseViewModel):
    _view_class = CorrelogramView
    _view_name = 'correlograms'

    binsize = None
    winsize_bins = None

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

        self.on_select(self.cluster_ids)

    def on_select(self, cluster_ids):
        super(CorrelogramViewModel, self).on_select(cluster_ids)
        spikes = self.spike_ids

        self.view.cluster_ids = cluster_ids

        # Compute the correlograms.
        spike_samples = self.model.spike_samples[spikes]
        spike_clusters = self.view.visual.spike_clusters

        ccgs = correlograms(spike_samples,
                            spike_clusters,
                            cluster_order=cluster_ids,
                            binsize=self.binsize,
                            winsize_bins=self.winsize_bins,
                            )
        ccgs = _symmetrize_correlograms(ccgs)
        # Normalize the CCGs.
        ccgs = ccgs * (1. / max(1., ccgs.max()))
        self.view.visual.correlograms = ccgs

        # Take the cluster order into account.
        self.view.visual.cluster_order = cluster_ids


class TraceViewModel(BaseViewModel):
    _view_class = TraceView
    _view_name = 'traces'
    _interval = None

    scale_factor = 1.
    n_samples_per_spike = 20
    interval_size = .25  # default interval size in milliseconds

    def _load_traces(self, interval):
        start, end = interval
        spikes = self.spike_ids

        # Load the traces.
        debug("Loading traces...")
        # Using channel_order ensures that we get rid of the dead channels.
        # We also keep the channel order as specified by the PRM file.
        traces = self.model.traces[start:end, self.model.channel_order]
        debug("Done!")

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
        # self._update_spike_clusters(spikes)
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
        self._interval = (start, end)
        self._load_traces((start, end))
        self.view.update()

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

    def on_key_press(self, event):
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

    def on_select(self, cluster_ids):
        # super(TraceViewModel, self).on_select(cluster_ids)
        self._selector.selected_clusters = cluster_ids
        # Get the spikes in the selected clusters.
        spikes = self.spike_ids
        n_clusters = len(cluster_ids)
        spike_clusters = self.model.spike_clusters[spikes]

        # Update the clusters of the trace view.
        visual = self._view.visual
        visual.spike_clusters = spike_clusters
        visual.cluster_ids = cluster_ids
        visual.cluster_order = cluster_ids
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
