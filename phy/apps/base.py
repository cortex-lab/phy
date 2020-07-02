# -*- coding: utf-8 -*-

"""Base controller to make clustering GUIs."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from functools import partial
import inspect
import logging
import os
from pathlib import Path
import shutil

import numpy as np
from scipy.signal import butter, lfilter

from phylib import _add_log_file
from phylib.io.array import SpikeSelector, _flatten
from phylib.stats import correlograms, firing_rate
from phylib.utils import Bunch, emit, connect, unconnect
from phylib.utils._misc import write_tsv

from phy.cluster._utils import RotatingProperty
from phy.cluster.supervisor import Supervisor
from phy.cluster.views.base import ManualClusteringView, BaseColorView
from phy.cluster.views import (
    WaveformView, FeatureView, TraceView, TraceImageView, CorrelogramView, AmplitudeView,
    ScatterView, ProbeView, RasterView, TemplateView, ISIView, FiringRateView, ClusterScatterView,
    select_traces)
from phy.cluster.views.trace import _iter_spike_waveforms
from phy.gui import GUI
from phy.gui.gui import _prompt_save
from phy.gui.qt import AsyncCaller
from phy.gui.state import _gui_state_path
from phy.gui.widgets import IPythonView
from phy.utils.context import Context, _cache_methods
from phy.utils.plugin import attach_plugins

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------------

def _concatenate_parents_attributes(cls, name):
    """Return the concatenation of class attributes of a given name among all parents of a
    class."""
    return _flatten([getattr(_, name, ()) for _ in inspect.getmro(cls)])


class Selection(Bunch):
    def __init__(self, controller):
        super(Selection, self).__init__()
        self.controller = controller

    @property
    def cluster_ids(self):
        return self.controller.supervisor.selected


class StatusBarHandler(logging.Handler):
    """Logging handler that displays messages in the status bar of a GUI."""

    def __init__(self, gui):
        self.gui = gui
        super(StatusBarHandler, self).__init__()

    def emit(self, record):
        self.gui.status_message = self.format(record)


#--------------------------------------------------------------------------
# Raw data filtering
#--------------------------------------------------------------------------

class RawDataFilter(RotatingProperty):
    def __init__(self):
        super(RawDataFilter, self).__init__()
        self.add('raw', lambda x, axis=None: x)

    def add_default_filter(self, sample_rate):
        b, a = butter(3, 150.0 / sample_rate * 2.0, 'high')

        @self.add_filter
        def high_pass(arr, axis=0):
            arr = lfilter(b, a, arr, axis=axis)
            arr = np.flip(arr, axis=axis)
            arr = lfilter(b, a, arr, axis=axis)
            arr = np.flip(arr, axis=axis)
            return arr
        self.set('high_pass')

    def add_filter(self, fun=None, name=None):
        """Add a raw data filter."""
        if fun is None:  # pragma: no cover
            return partial(self.add_filter, name=name)
        name = name or fun.__name__
        logger.debug("Add filter `%s`.", name)
        self.add(name, fun)

    def apply(self, arr, axis=None, name=None):
        """Filter raw data."""
        self.set(name or self.current)
        fun = self.get()
        if fun:
            logger.log(5, "Applying filter `%s` to raw data.", self.current)
            arrf = fun(arr, axis=axis)
            assert arrf.shape == arr.shape
            arr = arrf
        return arr


#------------------------------------------------------------------------------
# View mixins
#------------------------------------------------------------------------------

class WaveformMixin(object):
    n_spikes_waveforms = 100
    batch_size_waveforms = 10

    _state_params = (
        'n_spikes_waveforms', 'batch_size_waveforms',
    )

    _new_views = ('WaveformView',)

    # Map an amplitude type to a method name.
    _amplitude_functions = (
        ('raw', 'get_spike_raw_amplitudes'),
    )

    _waveform_functions = (
        ('waveforms', '_get_waveforms'),
        ('mean_waveforms', '_get_mean_waveforms'),
    )

    _cached = (
        # 'get_spike_raw_amplitudes',
        '_get_waveforms_with_n_spikes',
    )

    _memcached = (
        # 'get_mean_spike_raw_amplitudes',
        '_get_mean_waveforms',
    )

    def get_spike_raw_amplitudes(self, spike_ids, channel_id=None, **kwargs):
        """Return the maximum amplitude of the raw waveforms on the best channel of
        the first selected cluster.

        If `channel_id` is not specified, the returned amplitudes may be null.

        """
        # Spikes not kept get an amplitude of zero.
        out = np.zeros(len(spike_ids))
        # The cluster assignments of the requested spikes.
        spike_clusters = self.supervisor.clustering.spike_clusters[spike_ids]
        # Only keep spikes from clusters on the "best" channel.
        to_keep = np.in1d(spike_clusters, self.get_clusters_on_channel(channel_id))
        waveforms = self.model.get_waveforms(spike_ids[to_keep], [channel_id])
        if waveforms is not None:
            waveforms = waveforms[..., 0]
            assert waveforms.ndim == 2  # shape: (n_spikes_kept, n_samples)
            # Filter the waveforms.
            waveforms = self.raw_data_filter.apply(waveforms, axis=1)
            # Amplitudes of the kept spikes.
            amplitudes = waveforms.max(axis=1) - waveforms.min(axis=1)
            out[to_keep] = amplitudes
        assert np.all(out >= 0)
        return out

    def get_mean_spike_raw_amplitudes(self, cluster_id):
        """Return the average of the spike raw amplitudes."""
        spike_ids = self._get_amplitude_spike_ids(cluster_id)
        return np.mean(self.get_spike_raw_amplitudes(spike_ids))

    def _get_waveforms_with_n_spikes(
            self, cluster_id, n_spikes_waveforms, current_filter=None):

        # HACK: we pass self.raw_data_filter.current_filter so that it is cached properly.
        pos = self.model.channel_positions

        # Only keep spikes from the spike waveforms selection.
        if self.model.spike_waveforms is not None:
            subset_spikes = self.model.spike_waveforms.spike_ids
            spike_ids = self.selector(
                n_spikes_waveforms, [cluster_id], subset_spikes=subset_spikes)
        # Or keep spikes from a subset of the chunks for performance reasons (decompression will
        # happen on the fly here).
        else:
            spike_ids = self.selector(n_spikes_waveforms, [cluster_id], subset_chunks=True)

        # Get the best channels.
        channel_ids = self.get_best_channels(cluster_id)
        channel_labels = self._get_channel_labels(channel_ids)

        # Load the waveforms, either from the raw data directly, or from the _phy_spikes* files.
        data = self.model.get_waveforms(spike_ids, channel_ids)
        if data is not None:
            data = data - np.median(data, axis=1)[:, np.newaxis, :]
        assert data.ndim == 3  # n_spikes, n_samples, n_channels

        # Filter the waveforms.
        if data is not None:
            data = self.raw_data_filter.apply(data, axis=1)
        return Bunch(
            data=data,
            channel_ids=channel_ids,
            channel_labels=channel_labels,
            channel_positions=pos[channel_ids],
        )

    def _get_waveforms(self, cluster_id):
        """Return a selection of waveforms for a cluster."""
        return self._get_waveforms_with_n_spikes(
            cluster_id, self.n_spikes_waveforms, current_filter=self.raw_data_filter.current)

    def _get_mean_waveforms(self, cluster_id, current_filter=None):
        """Get the mean waveform of a cluster on its best channels."""
        b = self._get_waveforms(cluster_id)
        if b.data is not None:
            b.data = b.data.mean(axis=0)[np.newaxis, ...]
        b['alpha'] = 1.
        return b

    def _set_view_creator(self):
        super(WaveformMixin, self)._set_view_creator()
        self.view_creator['WaveformView'] = self.create_waveform_view

    def _get_waveforms_dict(self):
        waveform_functions = _concatenate_parents_attributes(
            self.__class__, '_waveform_functions')
        return {name: getattr(self, method) for name, method in waveform_functions}

    def create_waveform_view(self):
        waveforms_dict = self._get_waveforms_dict()
        if not waveforms_dict:
            return
        view = WaveformView(waveforms_dict, sample_rate=self.model.sample_rate)
        view.ex_status = self.raw_data_filter.current

        @connect(sender=view)
        def on_select_channel(sender, channel_id=None, key=None, button=None):
            # Update the Selection object with the channel id clicked in the waveform view.
            self.selection.channel_id = channel_id
            emit('selected_channel_changed', view)

        # Add extra actions.
        @connect(sender=view)
        def on_view_attached(view_, gui):
            # NOTE: this callback function is called in WaveformView.attach().

            @view.actions.add(
                alias='wn', prompt=True, prompt_default=lambda: str(self.n_spikes_waveforms))
            def change_n_spikes_waveforms(n_spikes_waveforms):
                """Change the number of spikes displayed in the waveform view."""
                self.n_spikes_waveforms = n_spikes_waveforms
                view.plot()

            view.actions.separator()

        @connect(sender=view)
        def on_close_view(view_, gui):
            unconnect(on_select_channel)
            unconnect(on_view_attached)

        return view


class FeatureMixin(object):
    n_spikes_features = 2500
    n_spikes_features_background = 2500

    _state_params = (
        'n_spikes_features', 'n_spikes_features_background',
    )

    _new_views = ('FeatureView',)

    _amplitude_functions = (
        ('feature', 'get_spike_feature_amplitudes'),
    )

    _cached = (
        '_get_features',
        'get_spike_feature_amplitudes',
    )

    def get_spike_feature_amplitudes(
            self, spike_ids, channel_id=None, channel_ids=None, pc=None, **kwargs):
        """Return the features for the specified channel and PC."""
        if self.model.features is None:
            return
        channel_id = channel_id if channel_id is not None else channel_ids[0]
        features = self._get_spike_features(spike_ids, [channel_id]).get('data', None)
        if features is None:  # pragma: no cover
            return
        assert features.shape[0] == len(spike_ids)
        logger.log(5, "Show channel %s and PC %s in amplitude view.", channel_id, pc)
        return features[:, 0, pc or 0]

    def create_amplitude_view(self):
        view = super(FeatureMixin, self).create_amplitude_view()
        if self.model.features is None:
            return view

        @connect
        def on_selected_feature_changed(sender):
            # Replot the amplitude view with the selected feature.
            view.amplitudes_type = 'feature'
            view.plot()

        @connect(sender=self.supervisor)
        def on_select(sender, cluster_ids, update_views=True):
            # Update the feature amplitude view when the cluster selection changes,
            # because the best channels change as well.
            if update_views and view.amplitudes_type == 'feature':
                view.plot()

        @connect(sender=view)
        def on_close_view(view_, gui):
            unconnect(on_selected_feature_changed)
            unconnect(on_select)

        return view

    def _get_feature_view_spike_ids(self, cluster_id=None, load_all=False):
        """Return some or all spikes belonging to a given cluster."""
        if cluster_id is None:
            spike_ids = self.get_background_spike_ids(self.n_spikes_features_background)
        # Compute features on the fly from spike waveforms.
        elif self.model.features is None and self.model.spike_waveforms is not None:
            spike_ids = self.get_spike_ids(cluster_id)
            assert len(spike_ids)
            spike_ids = np.intersect1d(spike_ids, self.model.spike_waveforms.spike_ids)
            if len(spike_ids) == 0:
                logger.debug("empty spikes for cluster %s", str(cluster_id))
            return spike_ids
        # Retrieve features from the self.model.features array.
        elif self.model.features is not None:
            # Load all spikes from the cluster if load_all is True.
            n = self.n_spikes_features if not load_all else None
            spike_ids = self.get_spike_ids(cluster_id, n=n)
        # Remove spike_ids that do not belong to model.features_rows
        if getattr(self.model, 'features_rows', None) is not None:  # pragma: no cover
            spike_ids = np.intersect1d(spike_ids, self.model.features_rows)
        return spike_ids

    def _get_feature_view_spike_times(self, cluster_id=None, load_all=False):
        """Return the times of some or all spikes belonging to a given cluster."""
        spike_ids = self._get_feature_view_spike_ids(cluster_id, load_all=load_all)
        if len(spike_ids) == 0:
            return
        spike_times = self._get_spike_times_reordered(spike_ids)
        return Bunch(
            data=spike_times,
            spike_ids=spike_ids,
            lim=(0., self.model.duration))

    def _get_spike_features(self, spike_ids, channel_ids):
        if len(spike_ids) == 0:  # pragma: no cover
            return Bunch()
        data = self.model.get_features(spike_ids, channel_ids)
        assert data.shape[:2] == (len(spike_ids), len(channel_ids))
        # Replace NaN values by zeros.
        data[np.isnan(data)] = 0
        assert data.shape[:2] == (len(spike_ids), len(channel_ids))
        assert np.isnan(data).sum() == 0
        channel_labels = self._get_channel_labels(channel_ids)
        return Bunch(
            data=data, spike_ids=spike_ids, channel_ids=channel_ids, channel_labels=channel_labels)

    def _get_features(self, cluster_id=None, channel_ids=None, load_all=False):
        """Return the features of a given cluster on specified channels."""
        spike_ids = self._get_feature_view_spike_ids(cluster_id, load_all=load_all)
        if len(spike_ids) == 0:  # pragma: no cover
            return Bunch()
        # Use the best channels only if a cluster is specified and
        # channels are not specified.
        if cluster_id is not None and channel_ids is None:
            channel_ids = self.get_best_channels(cluster_id)
        return self._get_spike_features(spike_ids, channel_ids)

    def create_feature_view(self):
        if self.model.features is None and getattr(self.model, 'spike_waveforms', None) is None:
            # NOTE: we can still construct the feature view when there are spike waveforms.
            return
        view = FeatureView(
            features=self._get_features,
            attributes={'time': self._get_feature_view_spike_times}
        )

        @connect
        def on_toggle_spike_reorder(sender, do_reorder):
            """Called when spike reordering is toggled."""
            self.selection.do_reorder = do_reorder
            view.plot()

        @connect(sender=view)
        def on_select_feature(sender, dim=None, channel_id=None, pc=None):
            # Update the Selection object with the channel id and PC clicked in the feature view.
            self.selection.channel_id = channel_id
            self.selection.feature_pc = pc
            emit('selected_feature_changed', view)

        connect(view.on_select_channel)
        connect(view.on_request_split)

        @connect(sender=view)
        def on_close_view(view_, gui):
            unconnect(on_toggle_spike_reorder)
            unconnect(on_select_feature)
            unconnect(view.on_select_channel)
            unconnect(view.on_request_split)

        return view

    def _set_view_creator(self):
        super(FeatureMixin, self)._set_view_creator()
        self.view_creator['FeatureView'] = self.create_feature_view


class TemplateMixin(object):
    """Support templates.

    The model needs to implement specific properties and methods.

    amplitudes : array-like
        The template amplitude of every spike (only with TemplateMixin).
    n_templates : int
        Initial number of templates.
    spike_templates : array-like
        The template initial id of every spike.
    get_template(template_id) : int => Bunch(template, channel_ids)
        Return the template data as a `(n_samples, n_channels)` array, the corresponding
        channel ids of the template.

    """

    _new_views = ('TemplateView',)

    _amplitude_functions = (
        ('template', 'get_spike_template_amplitudes'),
    )

    _waveform_functions = (
        ('templates', '_get_template_waveforms'),
    )

    _cached = (
        'get_amplitudes',
        'get_spike_template_amplitudes',
        'get_spike_template_features',
    )

    _memcached = (
        '_get_template_waveforms',
        'get_mean_spike_template_amplitudes',
        'get_template_counts',
        'get_template_for_cluster',
        'get_template_amplitude',
        'get_cluster_amplitude',
    )

    def __init__(self, *args, **kwargs):
        super(TemplateMixin, self).__init__(*args, **kwargs)

    def _get_amplitude_functions(self):
        out = super(TemplateMixin, self)._get_amplitude_functions()
        if getattr(self.model, 'template_features', None) is not None:
            out['template_feature'] = self.get_spike_template_features
        return out

    def get_amplitudes(self, cluster_id, load_all=False):
        """Return the spike amplitudes found in `amplitudes.npy`, for a given cluster."""
        spike_ids = self._get_amplitude_spike_ids(cluster_id, load_all=load_all)
        return self.model.amplitudes[spike_ids]

    def get_template_counts(self, cluster_id):
        """Return a histogram of the number of spikes in each template for a given cluster."""
        spike_ids = self.supervisor.clustering.spikes_per_cluster[cluster_id]
        st = self.model.spike_templates[spike_ids]
        return np.bincount(st, minlength=self.model.n_templates)

    def get_template_for_cluster(self, cluster_id):
        """Return the largest template associated to a cluster."""
        spike_ids = self.supervisor.clustering.spikes_per_cluster[cluster_id]
        st = self.model.spike_templates[spike_ids]
        template_ids, counts = np.unique(st, return_counts=True)
        ind = np.argmax(counts)
        return template_ids[ind]

    def get_template_amplitude(self, template_id):
        """Return the maximum amplitude of a template's waveforms across all channels."""
        waveforms = self.model.get_template(template_id).template
        assert waveforms.ndim == 2  # shape: (n_samples, n_channels)
        return (waveforms.max(axis=0) - waveforms.min(axis=0)).max()

    def get_cluster_amplitude(self, cluster_id):
        """Return the amplitude of the best template of a cluster."""
        template_id = self.get_template_for_cluster(cluster_id)
        return self.get_template_amplitude(template_id)

    def _set_cluster_metrics(self):
        """Add an amplitude column in the cluster view."""
        super(TemplateMixin, self)._set_cluster_metrics()
        self.cluster_metrics['amp'] = self.get_cluster_amplitude

    def get_spike_template_amplitudes(self, spike_ids, **kwargs):
        """Return the spike template amplitudes as stored in `amplitudes.npy`."""
        if self.model.amplitudes is None:
            return np.zeros(len(spike_ids))
        amplitudes = self.model.amplitudes[spike_ids]
        return amplitudes

    def get_spike_template_features(self, spike_ids, first_cluster=None, **kwargs):
        """Return the template features of the requested spikes onto the first selected
        cluster.

        This is "the dot product (projection) of each spike waveform onto the template of the
        first cluster."

        See @mswallac's comment at
        https://github.com/cortex-lab/phy/issues/868#issuecomment-520032905

        """
        assert first_cluster >= 0
        tf = self.model.get_template_features(spike_ids)
        if tf is None:
            return
        template = self.get_template_for_cluster(first_cluster)
        template_amplitudes = tf[:, template]
        assert template_amplitudes.shape == spike_ids.shape
        return template_amplitudes

    def get_mean_spike_template_amplitudes(self, cluster_id):
        """Return the average of the spike template amplitudes."""
        spike_ids = self._get_amplitude_spike_ids(cluster_id)
        return np.mean(self.get_spike_template_amplitudes(spike_ids))

    def _get_template_waveforms(self, cluster_id):
        """Return the waveforms of the templates corresponding to a cluster."""
        pos = self.model.channel_positions
        count = self.get_template_counts(cluster_id)
        template_ids = np.nonzero(count)[0]
        count = count[template_ids]
        # Get local channels.
        channel_ids = self.get_best_channels(cluster_id)
        # Get masks, related to the number of spikes per template which the cluster stems from.
        masks = count / float(count.max())
        masks = np.tile(masks.reshape((-1, 1)), (1, len(channel_ids)))
        # Get all templates from which this cluster stems from.
        templates = [self.model.get_template(template_id) for template_id in template_ids]
        # Construct the waveforms array.
        ns = self.model.n_samples_waveforms
        data = np.zeros((len(template_ids), ns, self.model.n_channels))
        for i, b in enumerate(templates):
            data[i][:, b.channel_ids] = b.template
        waveforms = data[..., channel_ids]
        assert waveforms.shape == (len(template_ids), ns, len(channel_ids))
        return Bunch(
            data=waveforms,
            channel_ids=channel_ids,
            channel_labels=self._get_channel_labels(channel_ids),
            channel_positions=pos[channel_ids],
            masks=masks, alpha=1.)

    def _get_all_templates(self, cluster_ids):
        """Get the template waveforms of a set of clusters."""
        out = {}
        for cluster_id in cluster_ids:
            waveforms = self._get_template_waveforms(cluster_id)
            out[cluster_id] = Bunch(
                template=waveforms.data[0, ...],
                channel_ids=waveforms.channel_ids,
            )
        return out

    def _set_view_creator(self):
        super(TemplateMixin, self)._set_view_creator()
        self.view_creator['TemplateView'] = self.create_template_view

    def create_template_view(self):
        """Create a template view."""
        view = TemplateView(
            templates=self._get_all_templates,
            channel_ids=np.arange(self.model.n_channels),
            channel_labels=self._get_channel_labels(),
        )
        self._attach_global_view(view)

        return view


class TraceMixin(object):

    _new_views = ('TraceView', 'TraceImageView')
    waveform_duration = 1.0  # in milliseconds

    def _get_traces(self, interval, show_all_spikes=False):
        """Get traces and spike waveforms."""
        traces_interval = select_traces(
            self.model.traces, interval, sample_rate=self.model.sample_rate)
        # Filter the loaded traces.
        traces_interval = self.raw_data_filter.apply(traces_interval, axis=0)
        out = Bunch(data=traces_interval)
        out.waveforms = list(_iter_spike_waveforms(
            interval=interval,
            traces_interval=traces_interval,
            model=self.model,
            supervisor=self.supervisor,
            n_samples_waveforms=int(round(1e-3 * self.waveform_duration * self.model.sample_rate)),
            get_best_channels=self.get_channel_amplitudes,
            show_all_spikes=show_all_spikes,
        ))
        return out

    def _trace_spike_times(self):
        cluster_ids = self.supervisor.selected
        if len(cluster_ids) == 0:
            return
        spc = self.supervisor.clustering.spikes_per_cluster
        spike_ids = spc[cluster_ids[0]]
        spike_times = self.model.spike_times[spike_ids]
        return spike_times

    def create_trace_view(self):
        """Create a trace view."""
        if self.model.traces is None:
            return

        view = TraceView(
            traces=self._get_traces,
            spike_times=self._trace_spike_times,
            sample_rate=self.model.sample_rate,
            duration=self.model.duration,
            n_channels=self.model.n_channels,
            channel_labels=self._get_channel_labels(),
            channel_positions=self.model.channel_positions,
        )

        # Update the get_traces() function with show_all_spikes.
        def _get_traces(interval):
            return self._get_traces(interval, show_all_spikes=view.show_all_spikes)
        view.traces = _get_traces
        view.ex_status = self.raw_data_filter.current

        @connect(sender=view)
        def on_select_spike(sender, channel_id=None, spike_id=None, cluster_id=None):
            # Update the global selection object.
            self.selection['spike_ids'] = [spike_id]
            # Select the corresponding cluster.
            self.supervisor.select([cluster_id])

        @connect
        def on_time_range_selected(sender, interval):
            self.selection['selected_time_range'] = interval

        @connect
        def on_select_time(sender, time):
            view.go_to(time)

        @connect(sender=view)
        def on_close_view(view_, gui):
            unconnect(on_select_spike)
            unconnect(on_time_range_selected)
            unconnect(on_select_time)

        return view

    def create_trace_image_view(self):
        """Create a trace image view."""
        if self.model.traces is None:
            return

        view = TraceImageView(
            traces=self._get_traces,
            sample_rate=self.model.sample_rate,
            duration=self.model.duration,
            n_channels=self.model.n_channels,
            channel_labels=self._get_channel_labels(),
            channel_positions=self.model.channel_positions,
        )

        @connect
        def on_select_time(sender, time):
            view.go_to(time)

        @connect(sender=view)
        def on_close_view(view_, gui):
            unconnect(on_select_time)

        return view

    def _set_view_creator(self):
        super(TraceMixin, self)._set_view_creator()
        self.view_creator['TraceView'] = self.create_trace_view
        self.view_creator['TraceImageView'] = self.create_trace_image_view


#------------------------------------------------------------------------------
# Base Controller
#------------------------------------------------------------------------------

class BaseController(object):
    """Base controller for manual clustering GUI.

    Constructor
    -----------
    dir_path : str or Path
        Path to the data directory
    config_dir : str or Path
        Path to the configuration directory
    model : Model
        Model object, optional (it is automatically created otherwise)
    plugins : list
        List of plugins to manually activate, optional (the plugins are automatically loaded from
        the user configuration directory).
    clear_cache : boolean
        Whether to clear the cache on startup.
    clear_state : boolean
        Whether to clear the GUI state files on startup.
    enable_threading : boolean
        Whether to enable threading in the views when selecting clusters.

    Methods to override
    -------------------

    The main methods that can be overriden when implementing a custom `Controller` are:

    _create_model() : None => object
        Return a Model instance (any object, see below) from the controller constructor's
        parameters.
    _set_view_creator() : None => None
        Populate the `self.view_creator` dictionary with custom views.
    get_best_channels(cluster_id) : int => list
        Return the list of best channels for any given cluster, sorted by decreasing match.

    Model
    -----

    The Model can be any object, but it needs to implement the following properties and methods
    in order to work with the BaseController:

    channel_mapping : array-like
        A `(n_channels,)` array with the column index in the raw data array of every channel.
        The displayed channel label of channel `channel_id` is `channel_mapping[channel_id]`.
    channel_positions : array-like
        A `(n_channels, 2)` array with the x, y coordinates of the electrode sites,
        in any unit (e.g. μm).
    channel_probes : array-like (optional)
        An `(n_channels,)` array with the probe index of every channel.
    channel_shanks : array-like (optional)
        An `(n_channels,)` array with the shank index of every channel (every probe might have
        multiple shanks). The shank index is relative to the probe. The pair (probe, shank)
        identifies uniquely a shank.
    duration : float
        The total duration of the recording, in seconds.
    features : array-like
        The object containing the features. The feature view is shown if this object is not None.
    metadata : dict
        Cluster metadata. Map metadata field names to dictionaries {cluster_id: value}.
        It is only expected to hold information representative of the state of the dataset
        on disk, not during a live clustering session.
        The special metadata field name `group` is reserved to cluster groups.
    n_channels : int
        Total number of channels in the recording (number of columns in the raw data array).
    n_samples_waveforms : int
        Number of time samples to use when extracting raw waveforms.
    sample_rate : float
        The sampling rate of the raw data.
    spike_attributes : dict
        Map attribute names to spike attributes, arrays of shape `(n_spikes,)`.
    spike_clusters : array-like
        Initial spike-cluster assignments, shape `(n_spikes,)`.
    spike_samples : array-like
        Spike samples, in samples, shape `(n_spikes,)`.
    spike_times : array-like
        Spike times, in seconds, shape `(n_spikes,)`.
    spike_waveforms : Bunch
        Extracted raw waveforms for a subset of the spikes.
        Should have attributes spike_ids, spike_channels, waveforms.
    traces : array-like
        Array (can be virtual/memmapped) of shape `(n_samples_total, n_channels)` with the
        raw data. The trace view is shown if this object is not None.

    get_features(spike_ids, channel_ids) : array-like, array-like => array-like
        Return spike features of specified spikes on the specified channels. Optional.
    get_waveforms(spike_ids, channel_ids) : array-like, array-like => array-like
        Return raw spike waveforms of specified spikes on the specified channels. Optional.

    save_spike_clusters(spike_clusters) : array-like => None
        Save spike clusters assignments back to disk.
    save_metadata(name, values) : str, dict => None
        Save cluster metadata, where name is the metadata field name, and values a dictionary
        `{cluster_id: value}`.

    Note
    ----

    The Model represents data as it is stored on disk. When cluster data changes during
    a manual clustering session (like spike-cluster assignments), the data in the model
    is not expected to change (it is rather the responsability of the controller).

    The model implements saving option for spike cluster assignments and cluster metadata.

    """

    gui_name = 'BaseGUI'
    gui_version = 2

    # Default value of the 'show_mapped_channels' param if it is not set in params.py.
    default_show_mapped_channels = True

    # Number of spikes to show in the views.
    n_spikes_amplitudes = 10000

    # Pairs (amplitude_type_name, method_name) where amplitude methods return spike amplitudes
    # of a given type.
    _amplitude_functions = (
    )

    n_spikes_correlograms = 100000

    # Number of raw data chunks to keep when loading waveforms from raw data (mostly useful
    # when using compressed dataset, as random access triggers expensive decompression).
    n_chunks_kept = 20

    # Controller attributes to load/save in the GUI state.
    _state_params = (
        'n_spikes_amplitudes', 'n_spikes_correlograms',
        'raw_data_filter_name',
    )

    # Methods that are cached in memory (and on disk) for performance.
    _memcached = (
        'get_mean_firing_rate',
        'get_best_channel',
        'get_best_channels',
        'get_channel_shank',
        'get_probe_depth',
        'peak_channel_similarity',
    )
    # Methods that are cached on disk for performance.
    _cached = (
        '_get_correlograms',
        '_get_correlograms_rate',
    )

    # Views to load by default.
    _new_views = (
        'ClusterScatterView', 'CorrelogramView', 'AmplitudeView',
        'ISIView', 'FiringRateView', 'ProbeView',
    )

    default_shortcuts = {
        'toggle_spike_reorder': 'ctrl+r',
        'switch_raw_data_filter': 'alt+r',
    }
    default_snippets = {}

    def __init__(
            self, dir_path=None, config_dir=None, model=None,
            clear_cache=None, clear_state=None,
            enable_threading=True, **kwargs):

        self._enable_threading = enable_threading

        assert dir_path
        self.dir_path = Path(dir_path).resolve()
        assert self.dir_path.exists()

        # Add a log file.
        _add_log_file(Path(dir_path) / 'phy.log')

        # Create or reuse a Model instance (any object)
        self.model = self._create_model(dir_path=dir_path, **kwargs) if model is None else model

        # Set up the cache.
        self._set_cache(clear_cache)

        # Raw data filter.
        self.raw_data_filter = RawDataFilter()
        self.raw_data_filter.add_default_filter(self.model.sample_rate)

        # Map view names to method creating new views. Other views can be added by plugins.
        self._set_view_creator()

        # Set default cluster metrics. Other metrics can be added by plugins.
        self._set_cluster_metrics()

        # Set the default similarity functions. Other similarity functions can be added by plugins.
        self._set_similarity_functions()

        # The controller.default_views can be set by the child class, otherwise it is computed
        # by concatenating all parents _new_views.
        if getattr(self, 'default_views', None) is None:
            self.default_views = _concatenate_parents_attributes(self.__class__, '_new_views')
        self._async_callers = {}
        self.config_dir = config_dir

        # Clear the GUI state files if needed.
        if clear_state:
            self._clear_state()

        self.selection = Selection(self)  # keep track of selected clusters, spikes, channels, etc.

        # Attach plugins before setting up the supervisor, so that plugins
        # can register callbacks to events raised during setup.
        # For example, 'request_cluster_metrics' to specify custom metrics
        # in the cluster and similarity views.
        self.attached_plugins = attach_plugins(
            self, config_dir=config_dir,
            plugins=kwargs.get('plugins', None), dirs=kwargs.get('plugin_dirs', None),
        )

        # Cache the methods specified in self._memcached and self._cached. All method names
        # are concatenated from the object's class parents and mixins.
        self._cache_methods()

        # Set up the Supervisor instance, responsible for the clustering process.
        self._set_supervisor()

        # Set up the Selector instance, responsible for selecting the spikes for display.
        self._set_selector()

        emit('controller_ready', self)

    # Internal initialization methods
    # -------------------------------------------------------------------------

    def _create_model(self, dir_path=None, **kwargs):
        """Create a model using the constructor parameters. To be overriden."""
        return

    def _clear_cache(self):
        logger.warn("Deleting the cache directory %s.", self.cache_dir)
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def _clear_state(self):
        """Clear the global and local GUI state files."""
        state_path = _gui_state_path(self.gui_name, config_dir=self.config_dir)
        if state_path.exists():
            logger.warning("Deleting %s.", state_path)
            state_path.unlink()
        local_path = self.cache_dir / 'state.json'
        if local_path.exists():
            local_path.unlink()
            logger.warning("Deleting %s.", local_path)

    def _set_cache(self, clear_cache=None):
        """Set up the cache, clear it if required, and create the Context instance."""
        self.cache_dir = self.dir_path / '.phy'
        if clear_cache:
            self._clear_cache()
        self.context = Context(self.cache_dir)

    def _set_view_creator(self):
        """Set the view creator, a dictionary mapping view names to methods creating views.

        May be overriden to add specific views.

        """
        self.view_creator = {
            'ClusterScatterView': self.create_cluster_scatter_view,
            'CorrelogramView': self.create_correlogram_view,
            'ISIView': self._make_histogram_view(ISIView, self._get_isi),
            'FiringRateView': self._make_histogram_view(FiringRateView, self._get_firing_rate),
            'AmplitudeView': self.create_amplitude_view,
            'ProbeView': self.create_probe_view,
            'RasterView': self.create_raster_view,
            'IPythonView': self.create_ipython_view,
        }
        # Spike attributes.
        for name, arr in getattr(self.model, 'spike_attributes', {}).items():
            view_name = 'Spike%sView' % name.title()
            self.view_creator[view_name] = self._make_spike_attributes_view(view_name, name, arr)

    def _set_cluster_metrics(self):
        """Set the cluster metrics dictionary with some default metrics."""
        self.cluster_metrics = {}  # dictionary {name: function cluster_id => value}, for plugins
        self.cluster_metrics['ch'] = self.get_best_channel_label
        if getattr(self.model, 'channel_shanks', None) is not None:
            self.cluster_metrics['sh'] = self.get_channel_shank
        self.cluster_metrics['depth'] = self.get_probe_depth
        self.cluster_metrics['fr'] = self.get_mean_firing_rate

    def _set_similarity_functions(self):
        """Set the `similarity_functions` dictionary that maps similarity names to functions
        `cluster_id => [(other_cluster_id, similarity_value)...]`."""
        self.similarity_functions = {
            'peak_channel': self.peak_channel_similarity,
        }
        # Default similarity function name.
        self.similarity = list(self.similarity_functions.keys())[0]

    def _set_supervisor(self):
        """Create the Supervisor instance."""
        # Load the new cluster id.
        new_cluster_id = self.context.load('new_cluster_id').get('new_cluster_id', None)

        # Cluster groups.
        cluster_groups = self.model.metadata.get('group', {})

        # Create the Supervisor instance.
        supervisor = Supervisor(
            spike_clusters=self.model.spike_clusters,
            cluster_groups=cluster_groups,
            cluster_metrics=self.cluster_metrics,
            cluster_labels=self.model.metadata,
            similarity=self.similarity_functions[self.similarity],
            new_cluster_id=new_cluster_id,
            context=self.context,
        )
        # Load the non-group metadata from the model to the cluster_meta.
        for name in sorted(self.model.metadata):
            if name == 'group':
                continue
            values = self.model.metadata.get(name, {})
            d = {cluster_id: {name: value} for cluster_id, value in values.items()}
            supervisor.cluster_meta.from_dict(d)

        # Connect the `save_clustering` event raised by the supervisor when saving
        # to the model's saving functions.
        connect(self.on_save_clustering, sender=supervisor)

        self.supervisor = supervisor

    def _set_selector(self):
        """Set the Selector instance."""

        def spikes_per_cluster(cluster_id):
            return self.supervisor.clustering.spikes_per_cluster.get(
                cluster_id, np.array([], dtype=np.int64))

        try:
            chunk_bounds = self.model.traces.chunk_bounds
        except AttributeError:
            chunk_bounds = [0.0, self.model.spike_samples[-1] + 1]

        self.selector = SpikeSelector(
            get_spikes_per_cluster=spikes_per_cluster,
            spike_times=self.model.spike_samples,  # NOTE: chunk_bounds is in samples, not seconds
            chunk_bounds=chunk_bounds,
            n_chunks_kept=self.n_chunks_kept)

    def _cache_methods(self):
        """Cache methods as specified in `self._memcached` and `self._cached`."""
        # Environment variable that can be used to disable the cache.
        if not os.environ.get('PHY_DISABLE_CACHE', False):
            memcached = _concatenate_parents_attributes(self.__class__, '_memcached')
            cached = _concatenate_parents_attributes(self.__class__, '_cached')
            _cache_methods(self, memcached, cached)

    def _get_channel_labels(self, channel_ids=None):
        """Return the labels of a list of channels."""
        if channel_ids is None:
            channel_ids = np.arange(self.model.n_channels)
        if (hasattr(self.model, 'channel_mapping') and
                getattr(self.model, 'show_mapped_channels', self.default_show_mapped_channels)):
            channel_labels = self.model.channel_mapping[channel_ids]
        else:
            channel_labels = channel_ids
        return ['%d' % ch for ch in channel_labels]

    # Internal view methods
    # -------------------------------------------------------------------------

    def _attach_global_view(self, view):
        """Attach a view deriving from BaseGlobalView.

        Make the view react to select, cluster, sort, filter events, color mapping, and make
        sure the view is populated at GUI startup, and when the view is added later.

        """

        # Async caller to avoid blocking cluster view loading when updating the view.
        # NOTE: it needs to be set as a property so as not to be garbage collected, leading
        # to Qt C++ segfaults.
        self._async_callers[view] = ac = AsyncCaller(delay=0)

        def resort(is_async=True, up=None):
            """Replot the view."""

            # Since we use the cluster ids in the order they appear in the cluster view, we
            # need to make sure that the cluster view is fully loaded.
            if not self.supervisor.cluster_view.is_ready():
                return

            def _update_plot():
                # The call to all_cluster_ids blocks until the cluster view JavaScript returns
                # the cluster ids.
                view.set_cluster_ids(self.supervisor.shown_cluster_ids)
                # Replot the view entirely.
                view.plot()
            if is_async:
                ac.set(_update_plot)
            else:
                # NOTE: we need to disable async after a clustering action, so that
                # the view needs to be properly updated *before* the newly created clusters
                # are selected.
                _update_plot()

        @connect(sender=self.supervisor.cluster_view)
        def on_table_sort(sender, cluster_ids):
            """Update the order of the clusters when the sort is changed in the cluster view."""
            if not view.auto_update or cluster_ids is None or not len(cluster_ids):
                return
            view.update_cluster_sort(cluster_ids)

        @connect(sender=self.supervisor.cluster_view)
        def on_table_filter(sender, cluster_ids):
            """Update the order of the clusters when a filtering is applied on the cluster view."""
            if not view.auto_update or cluster_ids is None or not len(cluster_ids):
                return
            view.set_cluster_ids(cluster_ids)
            view.plot()

        @connect(sender=self.supervisor)
        def on_cluster(sender, up):
            """Update the view after a clustering action."""
            if up.added:
                view.set_spike_clusters(self.supervisor.clustering.spike_clusters)
                if view.auto_update:
                    resort(is_async=False, up=up)

        connect(view.on_select)

        @connect(sender=view)
        def on_view_attached(view_, gui):
            # Populate the view when it is added to the GUI.
            resort()

        @connect(sender=self.supervisor.cluster_view)
        def on_ready(sender):
            """Populate the view at startup, as soon as the cluster view has been loaded."""
            resort()

        @connect(sender=view)
        def on_close_view(view_, gui):
            """Unconnect all events when closing the view."""
            unconnect(on_table_sort)
            unconnect(on_table_filter)
            unconnect(on_cluster)
            unconnect(view.on_select)
            unconnect(on_view_attached)
            unconnect(on_ready)

    # Saving methods
    # -------------------------------------------------------------------------

    def on_save_clustering(self, sender, spike_clusters, groups, *labels):
        """Save the modified data."""
        # Save the clusters.
        self.model.save_spike_clusters(spike_clusters)
        # Save cluster metadata.
        for name, values in labels:
            self.model.save_metadata(name, values)
        self._save_cluster_info()

    def _save_cluster_info(self):
        """Save all the contents of the cluster view into `cluster_info.tsv`."""
        # HACK: rename id to cluster_id for consistency in the cluster_info.tsv file.
        cluster_info = self.supervisor.cluster_info.copy()
        for d in cluster_info:
            d['cluster_id'] = d.pop('id')
        write_tsv(
            self.dir_path / 'cluster_info.tsv', cluster_info,
            first_field='cluster_id', exclude_fields=('is_masked',), n_significant_figures=8)

    # Model methods
    # -------------------------------------------------------------------------
    # These functions are defined here rather in the model, because they depend on the updated
    # spike-cluster assignments that change during manual clustering, whereas the model only
    # has initial spike-cluster assignments.

    def get_mean_firing_rate(self, cluster_id):
        """Return the mean firing rate of a cluster."""
        return self.supervisor.n_spikes(cluster_id) / max(1, self.model.duration)

    def get_best_channel(self, cluster_id):
        """Return the best channel id of a given cluster. This is the first channel returned
        by `get_best_channels()`."""
        channel_ids = self.get_best_channels(cluster_id)
        assert channel_ids is not None and len(channel_ids)
        return channel_ids[0]

    def get_best_channel_label(self, cluster_id):
        """Return the channel label of the best channel, for display in the cluster view."""
        return self._get_channel_labels([self.get_best_channel(cluster_id)])[0]

    def get_best_channels(self, cluster_id):  # pragma: no cover
        """Return the best channels of a given cluster. To be overriden."""
        logger.warning(
            "This method should be overriden and return a non-empty list of best channels.")
        return []

    def get_channel_amplitudes(self, cluster_id):  # pragma: no cover
        """Return the best channels of a given cluster along with their relative amplitudes.
        To be overriden."""
        logger.warning(
            "This method should be overriden.")
        return []

    def get_channel_shank(self, cluster_id):
        """Return the shank of a cluster's best channel, if the channel_shanks array is available.
        """
        best_channel_id = self.get_best_channel(cluster_id)
        return self.model.channel_shanks[best_channel_id]

    def get_probe_depth(self, cluster_id):
        """Return the depth of a cluster."""
        channel_id = self.get_best_channel(cluster_id)
        return 0 if channel_id is None else self.model.channel_positions[channel_id, 1]

    def get_clusters_on_channel(self, channel_id):
        """Return all clusters which have the specified channel among their best channels."""
        return [
            cluster_id for cluster_id in self.supervisor.clustering.cluster_ids
            if channel_id in self.get_best_channels(cluster_id)]

    # Default similarity functions
    # -------------------------------------------------------------------------

    def peak_channel_similarity(self, cluster_id):
        """Return the list of similar clusters to a given cluster, just on the basis of the
        peak channel.

        Parameters
        ----------
        cluster_id : int

        Returns
        -------
        similarities : list
            List of tuples `(other_cluster_id, similarity_value)` sorted by decreasing
            similarity value.

        """
        ch = self.get_best_channel(cluster_id)
        return [
            (other, 1.) for other in self.supervisor.clustering.cluster_ids
            if ch in self.get_best_channels(other)]

    # Public spike methods
    # -------------------------------------------------------------------------

    def get_spike_ids(self, cluster_id, n=None, **kwargs):
        """Return part or all of spike ids belonging to a given cluster."""
        return self.selector(n, [cluster_id], **kwargs)

    def get_spike_times(self, cluster_id, n=None):
        """Return the spike times of spikes returned by `get_spike_ids(cluster_id, n)`."""
        return self.model.spike_times[self.get_spike_ids(cluster_id, n=n)]

    def get_background_spike_ids(self, n=None):
        """Return regularly spaced spikes."""
        ns = len(self.model.spike_times)
        k = max(1, ns // n) if n is not None else 1
        return np.arange(0, ns, k)

    # Amplitudes
    # -------------------------------------------------------------------------

    def _get_spike_times_reordered(self, spike_ids):
        """Get spike times, reordered if needed."""
        spike_times = self.model.spike_times
        if (self.selection.get('do_reorder', None) and
                getattr(self.model, 'spike_times_reordered', None) is not None):
            spike_times = self.model.spike_times_reordered
        spike_times = spike_times[spike_ids]
        return spike_times

    def _get_amplitude_functions(self):
        """Return a dictionary mapping amplitude names to corresponding methods."""
        # Concatenation of all _amplitude_functions attributes in the class hierarchy.
        amplitude_functions = _concatenate_parents_attributes(
            self.__class__, '_amplitude_functions')
        return {name: getattr(self, method) for name, method in amplitude_functions}

    def _get_amplitude_spike_ids(self, cluster_id, load_all=False):
        """Return the spike ids for the amplitude view."""
        n = self.n_spikes_amplitudes if not load_all else None
        return self.get_spike_ids(cluster_id, n=n)

    def _amplitude_getter(self, cluster_ids, name=None, load_all=False):
        """Return the data requested by the amplitude view, wich depends on the
        type of amplitude.

        Parameters
        ----------
        cluster_ids : list
            List of clusters.
        name : str
            Amplitude name, see `self._amplitude_functions`.
        load_all : boolean
            Whether to load all spikes from the requested clusters, or a subselection just
            for display.

        """
        out = []
        n = self.n_spikes_amplitudes if not load_all else None
        # Find the first cluster, used to determine the best channels.
        first_cluster = next(cluster_id for cluster_id in cluster_ids if cluster_id is not None)
        # Best channels of the first cluster.
        channel_ids = self.get_best_channels(first_cluster)
        # Best channel of the first cluster.
        channel_id = channel_ids[0]
        # All clusters appearing on the first cluster's peak channel.
        other_clusters = self.get_clusters_on_channel(channel_id)
        # Get the amplitude method.
        f = self._get_amplitude_functions()[name]
        # Take spikes from the waveform selection if we're loading the raw amplitudes,
        # or by minimzing the number of chunks to load if fetching waveforms directly
        # from the raw data.
        # Otherwise we load the spikes randomly from the whole dataset.
        subset_chunks = subset_spikes = None
        if name == 'raw':
            if self.model.spike_waveforms is not None:
                subset_spikes = self.model.spike_waveforms.spike_ids
            else:
                subset_chunks = True
        # Go through each cluster in order to select spikes from each.
        for cluster_id in cluster_ids:
            if cluster_id is not None:
                # Cluster spikes.
                spike_ids = self.get_spike_ids(
                    cluster_id, n=n, subset_spikes=subset_spikes, subset_chunks=subset_chunks)
            else:
                # Background spikes.
                spike_ids = self.selector(
                    n, other_clusters, subset_spikes=subset_spikes, subset_chunks=subset_chunks)
            # Get the spike times.
            spike_times = self._get_spike_times_reordered(spike_ids)
            if name in ('feature', 'raw'):
                # Retrieve the feature PC selected in the feature view
                # or the channel selected in the waveform view.
                channel_id = self.selection.get('channel_id', channel_id)
            pc = self.selection.get('feature_pc', None)
            # Call the spike amplitude getter function.
            amplitudes = f(
                spike_ids, channel_ids=channel_ids, channel_id=channel_id, pc=pc,
                first_cluster=first_cluster)
            if amplitudes is None:
                continue
            assert amplitudes.shape == spike_ids.shape == spike_times.shape
            out.append(Bunch(
                amplitudes=amplitudes,
                spike_ids=spike_ids,
                spike_times=spike_times,
            ))
        return out

    def create_amplitude_view(self):
        """Create the amplitude view."""
        amplitudes_dict = {
            name: partial(self._amplitude_getter, name=name)
            for name in sorted(self._get_amplitude_functions())}
        if not amplitudes_dict:
            return
        # NOTE: we disable raw amplitudes for now as they're either too slow to load,
        # or they're loaded from a small part of the dataset which is not very useful.
        if len(amplitudes_dict) > 1 and 'raw' in amplitudes_dict:
            del amplitudes_dict['raw']
        view = AmplitudeView(
            amplitudes=amplitudes_dict,
            amplitudes_type=None,  # TODO: GUI state
            duration=self.model.duration,
        )

        @connect
        def on_toggle_spike_reorder(sender, do_reorder):
            """Called when spike reordering is toggled."""
            self.selection.do_reorder = do_reorder
            view.plot()

        @connect
        def on_selected_channel_changed(sender):
            """Called when a channel is selected in the waveform view."""
            # Do nothing if the displayed amplitude does not depend on the channel.
            if view.amplitudes_type not in ('feature', 'raw'):
                return
            # Otherwise, replot the amplitude view, which will use
            # Selection.selected_channel_id to use the requested channel in the computation of
            # the amplitudes.
            view.plot()

        @connect(sender=self.supervisor)
        def on_select(sender, cluster_ids, update_views=True):
            # Update the amplitude view when the cluster selection changes,
            # because the best channels change as well.
            if update_views and view.amplitudes_type == 'raw' and len(cluster_ids):
                # Update the channel used in the amplitude when the cluster selection changes.
                self.selection.channel_id = self.get_best_channel(cluster_ids[0])

        @connect
        def on_time_range_selected(sender, interval):
            # Show the time range in the amplitude view.
            view.show_time_range(interval)

        @connect(sender=view)
        def on_close_view(view_, gui):
            unconnect(on_toggle_spike_reorder)
            unconnect(on_selected_channel_changed)
            unconnect(on_select)
            unconnect(on_time_range_selected)

        return view

    # Cluster scatter view
    # -------------------------------------------------------------------------

    def create_cluster_scatter_view(self):
        """Create a cluster scatter view."""
        view = ClusterScatterView(
            cluster_ids=self.supervisor.clustering.cluster_ids,
            cluster_info=self.supervisor.get_cluster_info,
            # bindings={'x_axis': 'amp', 'y_axis': 'depth', 'size': 'fr'},
        )

        def _update():
            view.set_cluster_ids(self.supervisor.clustering.cluster_ids)
            view.plot()

        @connect(sender=self.supervisor.cluster_view)
        def on_table_filter(sender, cluster_ids):
            """Update the order of the clusters when a filtering is applied on the cluster view."""
            if not view.auto_update or cluster_ids is None or not len(cluster_ids):
                return
            view.set_cluster_ids(np.sort(cluster_ids))
            view.plot()

        @connect(sender=view)
        def on_view_attached(view_, gui):
            # Plot the view when adding it to the existing GUI.
            _update()

        @connect(sender=self.supervisor.cluster_view)
        def on_ready(sender):
            """Populate the view at startup, as soon as the cluster view has been loaded."""
            _update()

        @connect(sender=view)
        def on_close_view(view_, gui):
            """Unconnect all events when closing the view."""
            unconnect(on_table_filter)
            unconnect(on_view_attached)
            unconnect(on_ready)

        return view

    # Raster view
    # -------------------------------------------------------------------------

    def create_raster_view(self):
        """Create a raster view."""
        view = RasterView(
            self.model.spike_times,
            self.supervisor.clustering.spike_clusters,
            cluster_ids=self.supervisor.clustering.cluster_ids,
        )
        self._attach_global_view(view)

        return view

    # Correlograms
    # -------------------------------------------------------------------------

    def _get_correlograms(self, cluster_ids, bin_size, window_size):
        """Return the cross- and auto-correlograms of a set of clusters."""
        spike_ids = self.selector(self.n_spikes_correlograms, cluster_ids)
        st = self.model.spike_times[spike_ids]
        sc = self.supervisor.clustering.spike_clusters[spike_ids]
        return correlograms(
            st, sc, sample_rate=self.model.sample_rate, cluster_ids=cluster_ids,
            bin_size=bin_size, window_size=window_size)

    def _get_correlograms_rate(self, cluster_ids, bin_size):
        """Return the baseline firing rate of the cross- and auto-correlograms of clusters."""
        spike_ids = self.selector(self.n_spikes_correlograms, cluster_ids)
        sc = self.supervisor.clustering.spike_clusters[spike_ids]
        return firing_rate(
            sc, cluster_ids=cluster_ids, bin_size=bin_size, duration=self.model.duration)

    def create_correlogram_view(self):
        """Create a correlogram view."""
        return CorrelogramView(
            correlograms=self._get_correlograms,
            firing_rate=self._get_correlograms_rate,
            sample_rate=self.model.sample_rate,
        )

    # Probe view
    # -------------------------------------------------------------------------

    def create_probe_view(self):
        """Create a probe view."""
        return ProbeView(
            positions=self.model.channel_positions,
            best_channels=self.get_best_channels,
            channel_labels=self._get_channel_labels(),
        )

    # Histogram views
    # -------------------------------------------------------------------------

    def _make_histogram_view(self, view_cls, method):
        """Return a function that creates a HistogramView of a given class."""
        def _make():
            return view_cls(cluster_stat=method)
        return _make

    def _get_isi(self, cluster_id):
        """Return the ISI data of a cluster."""
        st = self.get_spike_times(cluster_id)
        intervals = np.diff(st)
        return Bunch(data=intervals)

    def _get_firing_rate(self, cluster_id):
        """Return the firing rate data of a cluster."""
        st = self.get_spike_times(cluster_id)
        dur = self.model.duration
        return Bunch(data=st, x_min=0, x_max=dur)

    # Spike attributes views
    # -------------------------------------------------------------------------

    def _make_spike_attributes_view(self, view_name, name, arr):
        """Create a special class deriving from ScatterView for each spike attribute."""
        def coords(cluster_ids, load_all=False):
            n = self.n_spikes_amplitudes if not load_all else None
            bunchs = []
            for cluster_id in cluster_ids:
                spike_ids = self.get_spike_ids(cluster_id, n=n)
                if arr.ndim == 1:
                    x = self.model.spike_times[spike_ids]
                    y = arr[spike_ids]
                    assert x.shape == y.shape == (len(spike_ids),)
                elif arr.ndim >= 2:
                    x, y = arr[spike_ids, :2].T
                bunchs.append(Bunch(x=x, y=y, spike_ids=spike_ids, data_bounds=None))
            return bunchs

        # Dynamic type deriving from ScatterView.
        view_cls = type(view_name, (ScatterView,), {})

        def _make():
            return view_cls(coords=coords)
        return _make

    # IPython View
    # -------------------------------------------------------------------------

    def create_ipython_view(self):
        """Create an IPython View."""
        view = IPythonView()
        view.start_kernel()
        view.inject(
            controller=self, c=self, m=self.model, s=self.supervisor,
            emit=emit, connect=connect,
        )
        return view

    # GUI
    # -------------------------------------------------------------------------

    def at_least_one_view(self, view_name):
        """Add a view of a given type if there is not already one.

        To be called before creating a GUI.

        """
        @connect(sender=self)
        def on_gui_ready(sender, gui):
            # Add a view automatically.
            if gui.view_count.get(view_name, 0) == 0:
                gui.create_and_add_view(view_name)

    def create_misc_actions(self, gui):

        # Toggle spike reorder.
        @gui.view_actions.add(
            shortcut=self.default_shortcuts['toggle_spike_reorder'],
            checkable=True, checked=False)
        def toggle_spike_reorder(checked):
            """Toggle spike time reordering."""
            logger.debug("%s spike time reordering.", 'Enable' if checked else 'Disable')
            emit('toggle_spike_reorder', self, checked)

        # Action to switch the raw data filter inthe trace and waveform views.
        @gui.view_actions.add(shortcut=self.default_shortcuts['switch_raw_data_filter'])
        def switch_raw_data_filter():
            """Switch the raw data filter."""
            filter_name = self.raw_data_filter.next()
            # Update the trace view.
            for v in gui.list_views(TraceView):
                if v.auto_update:
                    v.plot()
                    v.ex_status = filter_name
                    v.update_status()
            # Update the waveform view.
            for v in gui.list_views(WaveformView):
                if v.auto_update:
                    v.on_select_threaded(self.supervisor, self.supervisor.selected, gui=gui)
                    v.ex_status = filter_name
                    v.update_status()

        gui.view_actions.separator()

    def _add_default_color_schemes(self, view):
        """Add the default color schemes to every view."""
        group_colors = {
            'noise': 0,
            'mua': 1,
            'good': 2,
            None: 3,
            'unsorted': 3,
        }
        logger.debug("Adding default color schemes to %s.", view.name)

        def group_index(cluster_id):
            group = self.supervisor.cluster_meta.get('group', cluster_id)
            return group_colors.get(group, 0)  # TODO: better handling of colors for custom groups

        depth = self.supervisor.cluster_metrics['depth']
        fr = self.supervisor.cluster_metrics['fr']
        schemes = [
            # ('blank', 'blank', 0, False, False),
            ('random', 'categorical', lambda cl: cl, True, False),
            ('cluster_group', 'cluster_group', group_index, True, False),
            ('depth', 'linear', depth, False, False),
            ('firing_rate', 'linear', fr, False, True),
        ]
        for name, colormap, fun, categorical, logarithmic in schemes:
            view.add_color_scheme(
                name=name, fun=fun, cluster_ids=self.supervisor.clustering.cluster_ids,
                colormap=colormap, categorical=categorical, logarithmic=logarithmic)
        # Default color scheme.
        if not hasattr(view, 'color_scheme_name'):
            view.color_schemes.set('random')

    def create_gui(self, default_views=None, **kwargs):
        """Create the GUI.

        Constructor
        -----------

        default_views : list
            List of views to add in the GUI, optional. By default, all views from the view
            count are added.

        """
        default_views = self.default_views if default_views is None else default_views
        gui = GUI(
            name=self.gui_name,
            subtitle=str(self.dir_path),
            config_dir=self.config_dir,
            local_path=self.cache_dir / 'state.json',
            default_state_path=Path(inspect.getfile(self.__class__)).parent / 'static/state.json',
            view_creator=self.view_creator,
            default_views=default_views,
            enable_threading=self._enable_threading,
            **kwargs)

        # Set all state parameters from the GUI state.
        state_params = _concatenate_parents_attributes(self.__class__, '_state_params')
        for param in state_params:
            setattr(self, param, gui.state.get(param, getattr(self, param, None)))

        # Set the raw data filter from the GUI state.
        self.raw_data_filter.set(self.raw_data_filter_name)

        # Initial actions when creating views.
        @connect
        def on_view_attached(view, gui_):
            if gui_ != gui:
                return

            # Add default color schemes in each view.
            if isinstance(view, BaseColorView):
                self._add_default_color_schemes(view)

            if isinstance(view, ManualClusteringView):
                # Add auto update button.
                view.dock.add_button(
                    name='auto_update', icon='f021', checkable=True, checked=view.auto_update,
                    event='toggle_auto_update', callback=view.toggle_auto_update)

                # Show selected clusters when adding new views in the GUI.
                view.on_select(cluster_ids=self.supervisor.selected_clusters)

        # Get the state's current sort, and make sure the cluster view is initialized with it.
        self.supervisor.attach(gui)
        self.create_misc_actions(gui)
        gui.set_default_actions()
        gui.create_views()

        # Bind the `select_more` event to add clusters to the existing selection.
        @connect
        def on_select_more(sender, cluster_ids):
            self.supervisor.select(self.supervisor.selected + cluster_ids)

        @connect
        def on_request_select(sender, cluster_ids):
            self.supervisor.select(cluster_ids)

        # Prompt save.
        @connect(sender=gui)
        def on_close(sender):
            unconnect(on_view_attached, self)
            unconnect(on_select_more, self)
            unconnect(on_request_select, self)
            # Show save prompt if an action was done.
            do_prompt_save = kwargs.get('do_prompt_save', True)
            if do_prompt_save and self.supervisor.is_dirty():  # pragma: no cover
                r = _prompt_save()
                if r == 'save':
                    self.supervisor.save()
                elif r == 'cancel':
                    # Prevent closing of the GUI by returning False.
                    return False
                # Otherwise (r is 'close') we do nothing and close as usual.

        # Status bar handler
        handler = StatusBarHandler(gui)
        handler.setLevel(logging.INFO)
        logging.getLogger('phy').addHandler(handler)

        # Save the memcache when closing the GUI.
        @connect(sender=gui)  # noqa
        def on_close(sender):  # noqa

            # Gather all GUI state attributes from views that are local and thus need
            # to be saved in the data directory.
            for view in gui.views:
                local_keys = getattr(view, 'local_state_attrs', [])
                local_keys = ['%s.%s' % (view.name, key) for key in local_keys]
                gui.state.add_local_keys(local_keys)

            # Update the controller params in the GUI state.
            for param in self._state_params:
                gui.state[param] = getattr(self, param, None)

            # Save the memcache.
            gui.state['GUI_VERSION'] = self.gui_version
            self.context.save_memcache()

            # Remove the status bar handler when closing the GUI.
            logging.getLogger('phy').removeHandler(handler)

        try:
            emit('gui_ready', self, gui)
        except Exception as e:  # pragma: no cover
            logger.error(e)

        return gui
