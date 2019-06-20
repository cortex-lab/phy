# -*- coding: utf-8 -*-

"""Template GUI."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from functools import partial
import logging
from operator import itemgetter
import os
from pathlib import Path
import shutil

import numpy as np

from phylib.io.array import Selector, _index_of
from phylib.io.model import TemplateModel, get_template_params, load_model
from phylib.stats import correlograms, firing_rate
from phylib.utils import Bunch, emit, connect, unconnect
from phylib.utils._misc import write_tsv

from phy.cluster.supervisor import Supervisor
from phy.cluster.views.base import ManualClusteringView
from phy.cluster.views import (
    WaveformView, FeatureView, TraceView, CorrelogramView, AmplitudeView,
    ScatterView, ProbeView, RasterView, TemplateView, HistogramView, select_traces)
from phy.cluster.views.trace import _iter_spike_waveforms
from phy.gui import create_app, run_app, GUI
from phy.gui.gui import _prompt_save
from phy.gui.qt import AsyncCaller
from phy.gui.widgets import IPythonView
from phy.utils.context import Context, _cache_methods
from phy.utils.plugin import attach_plugins
from .. import _add_log_file

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Custom views
#------------------------------------------------------------------------------

class TemplateFeatureView(ScatterView):
    """Scatter view showing the template features."""
    pass


class ISIView(HistogramView):
    """Histogram view showing the interspike intervals."""
    x_max = .05  # window size is 50 ms by default
    n_bins = int(x_max / .001)  # by default, 1 bin = 1 ms
    alias_char = 'isi'  # provide `:isisn` (set number of bins) and `:isim` (set max bin) snippets


class FiringRateView(HistogramView):
    """Histogram view showing the time-dependent firing rate."""
    n_bins = 200
    alias_char = 'fr'


#------------------------------------------------------------------------------
# Template Controller
#------------------------------------------------------------------------------

class Selection(Bunch):
    def __init__(self, controller):
        super(Selection, self).__init__()
        self.controller = controller

    @property
    def cluster_ids(self):
        return self.controller.supervisor.selected

    @property
    def colormap(self):
        return self.controller.supervisor.color_selector.state.colormap

    @property
    def color_field(self):
        return self.controller.supervisor.color_selector.state.color_field


class TemplateController(object):
    """Controller for the Template GUI.

    Constructor
    -----------
    dat_path : str or Path or list
        Path to the raw data file(s)
    config_dir : str or Path
        Path to the configuration directory
    model : Model
        Model object, optional (it is automatically created otherwise)
    plugins : list
        List of plugins to manually activate, optional (the plugins are automatically loaded from
        the user configuration directory).

    """

    gui_name = 'TemplateGUI'

    # Number of spikes to show in the views.
    n_spikes_waveforms = 250
    batch_size_waveforms = 10
    n_spikes_features = 2500
    n_spikes_features_background = 2500
    n_spikes_amplitudes = 2500
    n_spikes_correlograms = 100000

    # Controller attributes to load/save in the GUI state.
    _state_params = (
        'n_spikes_waveforms', 'batch_size_waveforms', 'n_spikes_features',
        'n_spikes_features_background', 'n_spikes_amplitudes', 'n_spikes_correlograms')

    # Methods that are cached in memory (and on disk) for performance.
    _memcached = (
        'get_template_counts',
        'get_template_for_cluster',
        'get_mean_firing_rate',
        'get_best_channel',
        'get_best_channels',
        'get_channel_shank',
        'get_probe_depth',
        'get_template_amplitude',
        'get_mean_spike_template_amplitudes',
        'get_mean_spike_raw_amplitudes',
        '_get_template_waveforms',
    )
    # Methods that are cached on disk for performance.
    _cached = (
        'get_amplitudes',
        'get_spike_raw_amplitudes',
        'get_spike_template_amplitudes',
        '_get_spike_amplitudes',
        '_get_waveforms_with_n_spikes',
        '_get_features',
        '_get_feature_view_spike_times',
        '_get_template_features',
        '_get_correlograms',
        '_get_correlograms_rate',
    )

    # Views to load by default.
    _default_views = (
        'WaveformView', 'TraceView', 'FeatureView', 'TemplateFeatureView', 'CorrelogramView',
        'AmplitudeView', 'ISIView', 'FiringRateView'
    )

    def __init__(self, dat_path=None, config_dir=None, model=None, clear_cache=None, **kwargs):
        self.model = TemplateModel(dat_path, **kwargs) if not model else model
        self.cache_dir = self.model.dir_path / '.phy'
        # Clear the cache if needed.
        if clear_cache:
            logger.warn("Deleting the cache directory %s.", self.cache_dir)
            shutil.rmtree(self.cache_dir, ignore_errors=True)
        self.context = Context(self.cache_dir)
        self.config_dir = config_dir
        self._async_caller = AsyncCaller()
        self.selection = Selection(self)  # keep track of selected clusters, spikes, channels, etc.
        # mapping name => function {cluster_id: value}, to update in plugins
        self.cluster_metrics = {}
        self.view_creator = {
            'WaveformView': self.create_waveform_view,
            'TraceView': self.create_trace_view,
            'FeatureView': self.create_feature_view,
            'TemplateFeatureView': self.create_template_feature_view,
            'CorrelogramView': self.create_correlogram_view,
            'AmplitudeView': self.create_amplitude_view,
            'ProbeView': self.create_probe_view,
            'RasterView': self.create_raster_view,
            'TemplateView': self.create_template_view,
            'IPythonView': self.create_ipython_view,

            # Cluster statistics
            'ISIView': self._make_histogram_view(ISIView, self._get_isi),
            'FiringRateView': self._make_histogram_view(FiringRateView, self._get_firing_rate),
        }
        # Spike attributes.
        for name, arr in self.model.spike_attributes.items():
            view_name = 'Spike%sView' % name.title()
            self.view_creator[view_name] = self._make_spike_attributes_view(view_name, name, arr)

        self.default_views = list(self._default_views)

        # Attach plugins before setting up the supervisor, so that plugins
        # can register callbacks to events raised during setup.
        # For example, 'request_cluster_metrics' to specify custom metrics
        # in the cluster and similarity views.
        attach_plugins(self, plugins=kwargs.get('plugins', None), config_dir=config_dir)

        # Environment variable that can be used to disable the cache.
        if not os.environ.get('PHY_DISABLE_CACHE', False):
            _cache_methods(self, self._memcached, self._cached)
        self.supervisor = self._set_supervisor()
        self.selector = self._set_selector()

    # Internal methods
    # -------------------------------------------------------------------------

    def _set_supervisor(self):
        # Load the new cluster id.
        new_cluster_id = self.context.load('new_cluster_id').get('new_cluster_id', None)
        cluster_groups = self.model.get_metadata('group')
        # Special cluster metrics.
        cluster_metrics = {}
        cluster_metrics['channel'] = self.get_best_channel
        if self.model.channel_shanks is not None:
            cluster_metrics['shank'] = self.get_channel_shank
        cluster_metrics['depth'] = self.get_probe_depth
        cluster_metrics['firing_rate'] = self.get_mean_firing_rate
        # Add the controller's cluster metrics, could be coming from plugins.
        cluster_metrics.update(self.cluster_metrics)
        # Create the Supervisor instance.
        supervisor = Supervisor(
            spike_clusters=self.model.spike_clusters,
            cluster_groups=cluster_groups,
            cluster_metrics=cluster_metrics,
            cluster_labels=self.model.metadata,
            similarity=self.similarity,
            new_cluster_id=new_cluster_id,
            context=self.context,
        )
        # Load the non-group metadata from the model to the cluster_meta.
        for name in self.model.metadata_fields:
            if name == 'group':
                continue
            values = self.model.get_metadata(name)
            d = {cluster_id: {name: value} for cluster_id, value in values.items()}
            supervisor.cluster_meta.from_dict(d)

        @connect(sender=supervisor)
        def on_attach_gui(sender):
            @supervisor.actions.add(shortcut='shift+ctrl+k', set_busy=True)
            def split_init(cluster_ids=None):
                """Split a cluster according to the original templates."""
                if cluster_ids is None:
                    cluster_ids = supervisor.selected
                s = supervisor.clustering.spikes_in_clusters(cluster_ids)
                supervisor.actions.split(s, self.model.spike_templates[s])

            self.color_selector = supervisor.color_selector

        # Save.
        @connect(sender=supervisor)
        def on_save_clustering(sender, spike_clusters, groups, *labels):
            """Save the modified data."""
            # Save the clusters.
            self.model.save_spike_clusters(spike_clusters)
            # Save cluster metadata.
            for name, values in labels:
                self.model.save_metadata(name, values)
            # Save all the contents of the cluster view into cluster_info.tsv.
            write_tsv(
                self.model.dir_path / 'cluster_info.tsv', self.supervisor.cluster_info,
                first_field='id', exclude_fields=('is_masked',), n_significant_figures=8)

        return supervisor

    def _set_selector(self):
        def spikes_per_cluster(cluster_id):
            return self.supervisor.clustering.spikes_per_cluster.get(cluster_id, [0])
        return Selector(spikes_per_cluster)

    # Model methods
    # -------------------------------------------------------------------------

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

    def get_mean_firing_rate(self, cluster_id):
        """Return the mean firing rate of a cluster."""
        return "%.1f spk/s" % (self.supervisor.n_spikes(cluster_id) / max(1, self.model.duration))

    def get_best_channel(self, cluster_id):
        """Return the best channel of a given cluster."""
        template_id = self.get_template_for_cluster(cluster_id)
        return self.model.get_template(template_id).best_channel

    def get_best_channels(self, cluster_id):
        """Return the best channels of a given cluster."""
        template_id = self.get_template_for_cluster(cluster_id)
        return self.model.get_template(template_id).channel_ids

    def get_channel_shank(self, cluster_id):
        """Return the shank of a cluster's best channel, if the channel_shanks array is available.
        """
        best_channel_id = self.get_best_channel(cluster_id)
        return int(self.model.channel_shanks[best_channel_id])

    def get_probe_depth(self, cluster_id):
        """Return the depth of a cluster."""
        channel_id = self.get_best_channel(cluster_id)
        return self.model.channel_positions[channel_id][1]

    def get_clusters_on_channel(self, channel_id):
        """Return all clusters which have the specified channel among their best channels."""
        return [
            cluster_id for cluster_id in self.supervisor.clustering.cluster_ids
            if channel_id in self.get_best_channels(cluster_id)]

    def similarity(self, cluster_id):
        """Return the list of similar clusters to a given cluster."""
        # Templates of the cluster.
        temp_i = np.nonzero(self.get_template_counts(cluster_id))[0]
        # The similarity of the cluster with each template.
        sims = np.max(self.model.similar_templates[temp_i, :], axis=0)

        def _sim_ij(cj):
            # Templates of the cluster.
            if cj < self.model.n_templates:
                return float(sims[cj])
            temp_j = np.nonzero(self.get_template_counts(cj))[0]
            return float(np.max(sims[temp_j]))

        out = [(cj, _sim_ij(cj))
               for cj in self.supervisor.clustering.cluster_ids]
        # NOTE: hard-limit to 100 for performance reasons.
        return sorted(out, key=itemgetter(1), reverse=True)[:100]

    # Public spike methods
    # -------------------------------------------------------------------------

    def get_spike_ids(self, cluster_id, n=None):
        """Return part or all of spike ids belonging to a given cluster."""
        return self.selector.select_spikes([cluster_id], n)

    def get_spike_times(self, cluster_id, n=None):
        """Return the spike times of spikes returned by `get_spike_ids(cluster_id, n)`."""
        return self.model.spike_times[self.get_spike_ids(cluster_id, n=n)]

    def get_background_spike_ids(self, n=None):
        """Return regularly spaced spikes."""
        ns = self.model.n_spikes
        k = max(1, ns // n) if n is not None else 1
        return np.arange(0, ns, k)

    # Amplitudes
    # -------------------------------------------------------------------------

    def get_template_amplitude(self, template_id):
        """Return the maximum amplitude of a template's waveforms across all channels."""
        waveforms = self.model.get_template_waveforms(template_id)
        assert waveforms.ndim == 2  # shape: (n_samples, n_channels)
        return (waveforms.max(axis=0) - waveforms.min(axis=0)).max()

    def _get_amplitude_spike_ids(self, cluster_id, load_all=False):
        """Return the spike ids for the amplitude view."""
        n = self.n_spikes_amplitudes if not load_all else None
        return self.get_spike_ids(cluster_id, n=n)

    def _get_spike_amplitudes(
            self, spike_ids, name=None, channel_ids=None, channel_id=None, pc=None):
        """Return the requested type of amplitude, for the selected spikes."""
        if name is None:
            return self.model.amplitudes[spike_ids]
        elif name == 'template':
            amplitudes = self.model.amplitudes[spike_ids]
            # Spike-template assignments.
            spike_templates = self.model.spike_templates[spike_ids]
            # Find the template amplitudes of all templates appearing in spike_templates.
            unique_template_ids = np.unique(spike_templates)
            # Create an array with the template amplitudes.
            template_amplitudes = np.array(
                [self.get_template_amplitude(tid) for tid in unique_template_ids])
            # Get the template amplitude of every spike.
            spike_templates_rel = _index_of(spike_templates, unique_template_ids)
            assert spike_templates_rel.shape == amplitudes.shape
            # Multiply that by the spike amplitude.
            return template_amplitudes[spike_templates_rel] * amplitudes
        elif name == 'feature':
            # Return the features for the specified channel and PC.
            channel_id = channel_id if channel_id is not None else channel_ids[0]
            features = self._get_spike_features(spike_ids, [channel_id]).data
            return features[:, 0, pc or 0]
        elif name == 'raw':
            # WARNING: extracting raw waveforms is long!
            waveforms = self.model.get_waveforms(spike_ids, channel_ids)
            assert waveforms.ndim == 3  # shape: (n_spikes, n_samples, n_channels_loc)
            return (waveforms.max(axis=1) - waveforms.min(axis=1)).max(axis=1)

    def get_amplitudes(self, cluster_id, load_all=False):
        """Return the spike amplitudes found in `amplitudes.npy`, for a given cluster."""
        spike_ids = self._get_amplitude_spike_ids(cluster_id, load_all=load_all)
        return self._get_spike_amplitudes(spike_ids, name=None)

    def get_spike_raw_amplitudes(self, cluster_id, load_all=False):
        """Return the maximum amplitude of the raw waveforms across all channels."""
        spike_ids = self._get_amplitude_spike_ids(cluster_id, load_all=load_all)
        channel_ids = self.get_best_channels(cluster_id) if cluster_id is not None else None
        return self._get_spike_amplitudes(spike_ids, name='raw', channel_ids=channel_ids)

    def get_spike_template_amplitudes(self, cluster_id, load_all=False):
        """Return the template amplitudes multiplied by the spike's amplitude."""
        spike_ids = self._get_amplitude_spike_ids(cluster_id, load_all=load_all)
        return self._get_spike_amplitudes(spike_ids, name='template')

    def get_mean_spike_template_amplitudes(self, cluster_id):
        """Return the average of the spike template amplitudes."""
        return np.mean(self.get_spike_template_amplitudes(cluster_id))

    def get_mean_spike_raw_amplitudes(self, cluster_id):
        """Return the average of the spike raw amplitudes."""
        return np.mean(self.get_spike_raw_amplitudes(cluster_id))

    # Amplitude views
    # -------------------------------------------------------------------------

    def _amplitude_getter(self, cluster_ids, name=None, load_all=False):
        """Return the data requested by the amplitude view, wich depends on the
        type of amplitude."""
        out = []
        n = self.n_spikes_amplitudes if not load_all else None
        if name == 'raw' and n is not None:
            # HACK: currently extracting waveforms is very slow, we should probably save
            # a spike_waveforms.npy and spike_waveforms_ind.npy arrays.
            n //= 5
        # Find the first cluster, used to determine the best channels.
        first_cluster = next(cluster_id for cluster_id in cluster_ids if cluster_id is not None)
        channel_ids = self.get_best_channels(first_cluster)
        channel_id = channel_ids[0]
        # All clusters appearing on the first cluster's peak channel.
        other_clusters = self.get_clusters_on_channel(channel_id)
        for cluster_id in cluster_ids:
            if cluster_id is not None:
                # Cluster spikes.
                spike_ids = self.get_spike_ids(cluster_id, n=n)
            else:
                # Background spikes.
                spike_ids = self.selector.select_spikes(other_clusters, n)
            spike_times = self.model.spike_times[spike_ids]
            # Retrieve the feature PC selected in the feature view.
            # This is only used when name == 'feature'
            channel_id = self.selection.get('feature_channel_id', None)
            pc = self.selection.get('feature_pc', None)
            amplitudes = self._get_spike_amplitudes(
                spike_ids, name=name, channel_ids=channel_ids, channel_id=channel_id, pc=pc)
            out.append(Bunch(
                amplitudes=amplitudes,
                spike_ids=spike_ids,
                spike_times=spike_times,
            ))
        return out

    def create_amplitude_view(self):
        """Create the amplitude view."""
        amplitudes_dict = {
            'template': partial(self._amplitude_getter, name='template'),
            'feature': partial(self._amplitude_getter, name='feature'),
            'raw': partial(self._amplitude_getter, name='raw'),
        }
        view = AmplitudeView(
            amplitudes=amplitudes_dict,
            amplitude_name=None,  # TODO: GUI state
            duration=self.model.duration,
        )

        @connect
        def on_selected_feature_changed(sender):
            view.amplitude_name = 'feature'
            view.plot()

        @connect(sender=self.supervisor)
        def on_select(sender, cluster_ids, update_views=True):
            # Update the feature amplitude view when the cluster selection changes,
            # because the best channels change as well.
            if update_views and view.amplitude_name == 'feature':
                view.plot()

        return view

    # Waveforms
    # -------------------------------------------------------------------------

    def _get_waveforms_with_n_spikes(self, cluster_id, n_spikes_waveforms, batch_size_waveforms):
        pos = self.model.channel_positions
        spike_ids = self.selector.select_spikes(
            [cluster_id], n_spikes_waveforms, batch_size_waveforms)
        channel_ids = self.get_best_channels(cluster_id)
        data = self.model.get_waveforms(spike_ids, channel_ids)
        data = data - data.mean() if data is not None else None
        return Bunch(data=data, channel_ids=channel_ids, channel_positions=pos[channel_ids])

    def _get_waveforms(self, cluster_id):
        """Return a selection of waveforms for a cluster."""
        return self._get_waveforms_with_n_spikes(
            cluster_id, self.n_spikes_waveforms, self.batch_size_waveforms)

    def _get_mean_waveforms(self, cluster_id):
        """Get the mean waveform of a cluster on its best channels."""
        b = self._get_waveforms(cluster_id)
        if b.data is not None:
            b.data = b.data.mean(axis=0)[np.newaxis, ...]
        b['alpha'] = 1.
        return b

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
        # Get the mean amplitude for the cluster.
        mean_amp = self.get_amplitudes(cluster_id).mean()
        # Get all templates from which this cluster stems from.
        templates = [self.model.get_template(template_id) for template_id in template_ids]
        # Construct the waveforms array.
        ns = self.model.n_samples_templates
        data = np.zeros((len(template_ids), ns, self.model.n_channels))
        for i, b in enumerate(templates):
            data[i][:, b.channel_ids] = b.template * mean_amp
        waveforms = data[..., channel_ids]
        assert waveforms.shape == (len(template_ids), ns, len(channel_ids))
        return Bunch(
            data=waveforms, channel_ids=channel_ids, channel_positions=pos[channel_ids],
            masks=masks, alpha=1.)

    def create_waveform_view(self):
        f = (self._get_waveforms if self.model.traces is not None
             else self._get_template_waveforms)
        v = WaveformView(waveforms=f)
        v.shortcuts['toggle_templates'] = 'w'
        v.shortcuts['toggle_mean_waveforms'] = 'm'

        v.state_attrs += ('show_what',)
        funs = {
            'waveforms': self._get_waveforms,
            'templates': self._get_template_waveforms,
            'mean_waveforms': self._get_mean_waveforms,
        }

        # Add extra actions.
        @connect(sender=v)
        def on_view_actions_created(sender):
            # NOTE: this callback function is called in WaveformView.attach().

            # Initialize show_what if it was not set in the GUI state.
            if not hasattr(v, 'show_what'):  # pragma: no cover
                v.show_what = 'waveforms'
            # Set the waveforms function.
            v.waveforms = funs[v.show_what]

            @v.actions.add(checkable=True, checked=v.show_what == 'templates')
            def toggle_templates(checked):
                """Show templates instead of spike waveforms."""
                # Both checkboxes are mutually exclusive.
                if checked:
                    v.actions.get('toggle_mean_waveforms').setChecked(False)
                v.show_what = 'templates' if checked else 'waveforms'
                if v.show_what == 'waveforms' and self.model.traces is None:
                    return
                v.waveforms = funs[v.show_what]
                v.on_select(cluster_ids=v.cluster_ids)

            @v.actions.add(checkable=True, checked=v.show_what == 'mean_waveforms')
            def toggle_mean_waveforms(checked):
                """Show mean waveforms instead of spike waveforms."""
                # Both checkboxes are mutually exclusive.
                if checked:
                    v.actions.get('toggle_templates').setChecked(False)
                v.show_what = 'mean_waveforms' if checked else 'waveforms'
                v.waveforms = funs[v.show_what]
                v.on_select(cluster_ids=v.cluster_ids)

            @v.actions.add(
                alias='wn', prompt=True, prompt_default=lambda: str(self.n_spikes_waveforms))
            def change_n_spikes_waveforms(n_spikes_waveforms):
                """Change the number of spikes displayed in the waveform view."""
                self.n_spikes_waveforms = n_spikes_waveforms
                v.on_select(cluster_ids=v.cluster_ids)

            v.actions.separator()

        return v

    # Template view
    # -------------------------------------------------------------------------

    def _get_all_templates(self, cluster_ids):
        """Get the template waveforms of a set of clusters."""
        bunchs = {
            cluster_id: self._get_template_waveforms(cluster_id)
            for cluster_id in cluster_ids}
        mean_amp = {
            cluster_id: self.get_amplitudes(cluster_id).mean()
            for cluster_id in cluster_ids}
        return {
            cluster_id: Bunch(
                template=bunchs[cluster_id].data[0, ...] * mean_amp[cluster_id],
                channel_ids=bunchs[cluster_id].channel_ids)
            for cluster_id in cluster_ids}

    def create_template_view(self):
        """Create a template view."""
        view = TemplateView(
            templates=self._get_all_templates,
            channel_ids=np.arange(self.model.n_channels),
            cluster_color_selector=self.color_selector,
        )

        @connect(sender=view)
        def on_cluster_click(sender, cluster_id, key=None, button=None):
            self.supervisor.select([cluster_id])

        @connect(sender=self.supervisor)
        def on_cluster(sender, up):
            if view.auto_update and up.added:
                view.set_cluster_ids(self.supervisor.clustering.cluster_ids)
                view.plot()

        @connect(sender=self.supervisor.cluster_view)
        def on_table_sort(sender, cluster_ids):
            if not view.auto_update:
                return
            view.update_cluster_sort(cluster_ids)

        @connect(sender=self.supervisor.cluster_view)
        def on_table_filter(sender, cluster_ids):
            if not view.auto_update or cluster_ids is None or not len(cluster_ids):
                return
            view.set_cluster_ids(cluster_ids)
            view.plot()

        @connect(sender=self.supervisor)
        def on_color_mapping_changed(sender):
            view.update_color(self.supervisor.selected_clusters)

        @connect
        def on_close_view(sender, view_):
            if view_ == view:
                unconnect(on_table_filter)
                unconnect(on_table_sort)
                unconnect(on_color_mapping_changed)

        # Initial sort.
        @connect(sender=self.supervisor.cluster_view)
        def on_ready(sender):
            @self.supervisor.cluster_view.get_ids
            def init_cluster_ids(cluster_ids):
                assert cluster_ids is not None
                view.set_cluster_ids(cluster_ids)
                @self._async_caller.set
                def _update_plot():
                    view.plot()

        return view

    # Features
    # -------------------------------------------------------------------------

    def _get_feature_view_spike_ids(self, cluster_id=None, load_all=False):
        """Return some or all spikes belonging to a given cluster."""
        if cluster_id is None:
            spike_ids = self.get_background_spike_ids(self.n_spikes_features_background)
        else:
            # Load all spikes from the cluster if load_all is True.
            n = self.n_spikes_features if not load_all else None
            spike_ids = self.get_spike_ids(cluster_id, n=n)
        # Remove spike_ids that do not belong to model.features_rows
        if self.model.features_rows is not None:  # pragma: no cover
            spike_ids = np.intersect1d(spike_ids, self.model.features_rows)
        return spike_ids

    def _get_feature_view_spike_times(self, cluster_id=None, load_all=False):
        """Return the times of some or all spikes belonging to a given cluster."""
        spike_ids = self._get_feature_view_spike_ids(cluster_id, load_all=load_all)
        return Bunch(
            data=self.model.spike_times[spike_ids],
            spike_ids=spike_ids,
            lim=(0., self.model.duration))

    def _get_spike_features(self, spike_ids, channel_ids):
        data = self.model.get_features(spike_ids, channel_ids)
        assert data.shape[:2] == (len(spike_ids), len(channel_ids))
        # Remove rows with at least one nan value.
        nan = np.unique(np.nonzero(np.isnan(data))[0])
        nonnan = np.setdiff1d(np.arange(len(spike_ids)), nan)
        data = data[nonnan, ...]
        spike_ids = spike_ids[nonnan]
        assert data.shape[:2] == (len(spike_ids), len(channel_ids))
        assert np.isnan(data).sum() == 0
        return Bunch(data=data, spike_ids=spike_ids, channel_ids=channel_ids)

    def _get_features(self, cluster_id=None, channel_ids=None, load_all=False):
        """Return the features of a given cluster on specified channels."""
        spike_ids = self._get_feature_view_spike_ids(cluster_id, load_all=load_all)
        # Use the best channels only if a cluster is specified and
        # channels are not specified.
        if cluster_id is not None and channel_ids is None:
            channel_ids = self.get_best_channels(cluster_id)
        return self._get_spike_features(spike_ids, channel_ids)

    def create_feature_view(self):
        if self.model.features is None:
            return
        view = FeatureView(
            features=self._get_features,
            attributes={'time': self._get_feature_view_spike_times}
        )

        @connect(sender=view)
        def on_feature_click(sender, dim=None, channel_id=None, pc=None):
            # Update the Selection object with the channel id and PC clicked in the feature view.
            self.selection.feature_channel_id = channel_id
            self.selection.feature_pc = pc
            emit('selected_feature_changed', view)

        return view

    # Template features
    # -------------------------------------------------------------------------

    def _get_template_features(self, cluster_ids, load_all=False):
        """Get the template features of a pair of clusters."""
        if len(cluster_ids) != 2:
            return
        assert len(cluster_ids) == 2
        clu0, clu1 = cluster_ids

        s0 = self._get_feature_view_spike_ids(clu0, load_all=load_all)
        s1 = self._get_feature_view_spike_ids(clu1, load_all=load_all)

        n0 = self.get_template_counts(clu0)
        n1 = self.get_template_counts(clu1)

        t0 = self.model.get_template_features(s0)
        t1 = self.model.get_template_features(s1)

        x0 = np.average(t0, weights=n0, axis=1)
        y0 = np.average(t0, weights=n1, axis=1)

        x1 = np.average(t1, weights=n0, axis=1)
        y1 = np.average(t1, weights=n1, axis=1)

        return [
            Bunch(x=x0, y=y0, spike_ids=s0),
            Bunch(x=x1, y=y1, spike_ids=s1),
        ]

    def create_template_feature_view(self):
        if self.model.template_features is None:
            return
        return TemplateFeatureView(coords=self._get_template_features)

    # Traces
    # -------------------------------------------------------------------------

    def _get_traces(self, interval, show_all_spikes=False):
        """Get traces and spike waveforms."""
        k = self.model.n_samples_templates
        m = self.model

        traces_interval = select_traces(m.traces, interval, sample_rate=m.sample_rate)
        # Reorder vertically.
        out = Bunch(data=traces_interval)

        def gbc(cluster_id):
            return self.get_best_channels(cluster_id)

        out.waveforms = list(_iter_spike_waveforms(
            interval=interval,
            traces_interval=traces_interval,
            model=self.model,
            supervisor=self.supervisor,
            color_selector=self.color_selector,
            n_samples_waveforms=k,
            get_best_channels=gbc,
            show_all_spikes=show_all_spikes,
        ))
        return out

    def _trace_spike_times(self):
        m = self.model
        cluster_ids = self.supervisor.selected
        if len(cluster_ids) == 0:
            return
        spc = self.supervisor.clustering.spikes_per_cluster
        spike_ids = spc[cluster_ids[0]]
        spike_times = m.spike_times[spike_ids]
        return spike_times

    def create_trace_view(self):
        """Create a trace view."""
        if self.model.traces is None:
            return

        m = self.model
        v = TraceView(
            traces=self._get_traces,
            spike_times=self._trace_spike_times,
            n_channels=m.n_channels,
            sample_rate=m.sample_rate,
            duration=m.duration,
            channel_vertical_order=m.channel_vertical_order,
        )

        # Update the get_traces() function with show_all_spikes.
        def _get_traces(interval):
            return self._get_traces(interval, show_all_spikes=v.show_all_spikes)
        v.traces = _get_traces

        @connect(sender=v)
        def on_spike_click(sender, channel_id=None, spike_id=None, cluster_id=None):
            self.selection['spike_ids'] = [spike_id]

        @connect(sender=v)  # noqa
        def on_spike_click(sender, channel_id=None, spike_id=None, cluster_id=None):
            # Select the corresponding cluster.
            self.supervisor.select([cluster_id])
            # Update the trace view.
            v.on_select([cluster_id])

        @connect(sender=self.supervisor)
        def on_color_mapping_changed(sender):
            v.on_select()

        @connect
        def on_close_view(sender, view):
            if view == v:
                unconnect(on_spike_click)
                unconnect(on_color_mapping_changed)

        return v

    # Correlograms
    # -------------------------------------------------------------------------

    def _get_correlograms(self, cluster_ids, bin_size, window_size):
        """Return the cross- and auto-correlograms of a set of clusters."""
        spike_ids = self.selector.select_spikes(
            cluster_ids, self.n_spikes_correlograms, subset='random')
        st = self.model.spike_times[spike_ids]
        sc = self.supervisor.clustering.spike_clusters[spike_ids]
        return correlograms(
            st, sc, sample_rate=self.model.sample_rate, cluster_ids=cluster_ids,
            bin_size=bin_size, window_size=window_size)

    def _get_correlograms_rate(self, cluster_ids, bin_size):
        """Return the baseline firing rate of the cross- and auto-correlograms of clusters."""
        spike_ids = self.selector.select_spikes(
            cluster_ids, self.n_spikes_correlograms, subset='random')
        sc = self.supervisor.clustering.spike_clusters[spike_ids]
        return firing_rate(
            sc, cluster_ids=cluster_ids, bin_size=bin_size, duration=self.model.duration)

    def create_correlogram_view(self):
        """Create a correlogram view."""
        m = self.model
        return CorrelogramView(
            correlograms=self._get_correlograms,
            firing_rate=self._get_correlograms_rate,
            sample_rate=m.sample_rate,
        )

    # Probe view
    # -------------------------------------------------------------------------

    def create_probe_view(self):
        """Create a probe view."""
        return ProbeView(
            positions=self.model.channel_positions,
            best_channels=self.get_best_channels,
        )

    # Raster view
    # -------------------------------------------------------------------------

    def create_raster_view(self):
        """Create a raster view."""
        view = RasterView(
            self.model.spike_times,
            self.supervisor.clustering.spike_clusters,
            cluster_color_selector=self.color_selector,
        )

        @connect(sender=view)
        def on_cluster_click(sender, cluster_id, key=None, button=None):
            self.supervisor.select([cluster_id])

        @connect(sender=self.supervisor)
        def on_cluster(sender, up):
            if view.auto_update and up.added:
                view.set_spike_clusters(self.supervisor.clustering.spike_clusters)
                view.set_cluster_ids(self.supervisor.clustering.cluster_ids)
                view.plot()

        @connect(sender=self.supervisor.cluster_view)
        def on_table_sort(sender, cluster_ids):
            if not view.auto_update or cluster_ids is None or not len(cluster_ids):
                return
            # OPTIM: do not need to replot everything, but just to change the ordering)
            view.update_cluster_sort(cluster_ids)

        @connect(sender=self.supervisor.cluster_view)
        def on_table_filter(sender, cluster_ids):
            if not view.auto_update or cluster_ids is None or not len(cluster_ids):
                return
            view.set_cluster_ids(cluster_ids)
            view.plot()

        @connect(sender=self.supervisor)
        def on_color_mapping_changed(sender):
            view.update_color(self.supervisor.selected_clusters)

        @connect
        def on_close_view(sender, view_):
            if view_ == view:
                unconnect(on_table_sort)
                unconnect(on_table_filter)
                unconnect(on_color_mapping_changed)

        # Initial sort.
        @connect(sender=self.supervisor.cluster_view)
        def on_ready(sender):
            @self.supervisor.cluster_view.get_ids
            def init_cluster_ids(cluster_ids):
                assert cluster_ids is not None
                view.set_cluster_ids(cluster_ids)
                view.plot()

        return view

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
        return Bunch(data=st, x_max=dur)

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
        view.inject(controller=self, c=self, m=self.model, s=self.supervisor)
        return view

    # GUI
    # -------------------------------------------------------------------------

    def create_gui(self, default_views=None, **kwargs):
        """Create the template GUI.

        Constructor
        -----------

        default_views : list
            List of views to add in the GUI, optional. By default, all views from the view
            count are added.

        """
        default_views = self.default_views if default_views is None else default_views
        gui = GUI(
            name=self.gui_name,
            subtitle=str(self.model.dir_path),
            config_dir=self.config_dir,
            local_path=self.cache_dir / 'state.json',
            default_state_path=Path(__file__).parent / 'static/state.json',
            view_creator=self.view_creator,
            default_views=default_views,
            **kwargs)

        # If the n_spikes_* parameters are set in the GUI state, load them in the controller.
        for param in self._state_params:
            setattr(self, param, gui.state.get(param, getattr(self, param, None)))

        # Get the state's current sort, and make sure the cluster view is initialized with it.
        self.supervisor.attach(gui)

        gui.set_default_actions()
        gui.create_views()

        @connect(sender=gui)
        def on_add_view(sender, view):
            if isinstance(view, ManualClusteringView):
                view.on_select(cluster_ids=self.supervisor.selected_clusters)

        # Save the memcache when closing the GUI.
        @connect(sender=gui)
        def on_close(sender):
            # Show save prompt if an action was done.
            if self.supervisor.is_dirty():  # pragma: no cover
                r = _prompt_save()
                if r == 'save':
                    self.supervisor.save()
                elif r == 'cancel':
                    # Prevent closing of the GUI by returning False.
                    return False
                # Otherwise (r is 'close') we do nothing and close as usual.
            unconnect(on_add_view)

            # Gather all GUI state attributes from views that are local and thus need
            # to be saved in the data directory.
            gui.state._local_keys = set().union(
                *(getattr(view, 'local_state_attrs', ()) for view in gui.views))

            # Update the controller params in the GUI state.
            for param in self._state_params:
                gui.state[param] = getattr(self, param, None)

            self.context.save_memcache()

        emit('gui_ready', self, gui)

        return gui


#------------------------------------------------------------------------------
# Template commands
#------------------------------------------------------------------------------

def template_gui(params_path, clear_cache=None):  # pragma: no cover
    """Launch the Template GUI."""
    # Create a `phy.log` log file with DEBUG level.
    _add_log_file(Path(params_path).parent / 'phy.log')

    create_app()
    controller = TemplateController(**get_template_params(params_path), clear_cache=clear_cache)
    gui = controller.create_gui()
    gui.show()
    run_app()
    gui.close()


def template_describe(params_path):
    """Describe a template dataset."""
    load_model(params_path).describe()
