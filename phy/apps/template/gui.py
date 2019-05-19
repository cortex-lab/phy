# -*- coding: utf-8 -*-

"""Template GUI."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
from operator import itemgetter
import os
import os.path as op

import numpy as np

from phylib.io.array import Selector
from phylib.io.model import TemplateModel
from phylib.stats import correlograms, firing_rate
from phylib.utils import Bunch, emit, connect, unconnect
from phylib.utils._color import ColorSelector
from phylib.utils._misc import _read_python

from phy.cluster.supervisor import Supervisor
from phy.cluster.views import (WaveformView,
                               FeatureView,
                               TraceView,
                               CorrelogramView,
                               ScatterView,
                               ProbeView,
                               select_traces,
                               )
from phy.cluster.views.trace import _iter_spike_waveforms
from phy.gui import create_app, run_app, GUI
from phy.utils.context import Context, _cache_methods
from phy.utils.plugin import attach_plugins
from .. import _add_log_file

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Utils and views
#------------------------------------------------------------------------------

class TemplateFeatureView(ScatterView):
    def _get_data(self, cluster_ids):
        if len(cluster_ids) != 2:
            return []
        b = self.coords(cluster_ids)
        return [Bunch(x=b.x0, y=b.y0), Bunch(x=b.x1, y=b.y1)]


class AmplitudeView(ScatterView):
    _default_position = 'right'


#------------------------------------------------------------------------------
# Template Controller
#------------------------------------------------------------------------------

class TemplateController(object):
    gui_name = 'TemplateGUI'

    n_spikes_waveforms = 100
    batch_size_waveforms = 10

    n_spikes_features = 2500
    n_spikes_amplitudes = 5000
    n_spikes_correlograms = 100000

    def __init__(self, dat_path=None, config_dir=None, model=None, **kwargs):
        if model is None:
            assert dat_path
            dat_path = op.abspath(dat_path)
            self.model = TemplateModel(dat_path, **kwargs)
        else:
            self.model = model
        self.cache_dir = op.join(self.model.dir_path, '.phy')
        self.context = Context(self.cache_dir)
        self.config_dir = config_dir
        self.view_creator = {
            WaveformView: self.create_waveform_view,
            TraceView: self.create_trace_view,
            FeatureView: self.create_feature_view,
            TemplateFeatureView: self.create_template_feature_view,
            CorrelogramView: self.create_correlogram_view,
            AmplitudeView: self.create_amplitude_view,
            ProbeView: self.create_probe_view,
        }

        # Attach plugins before setting up the supervisor, so that plugins
        # can register callbacks to events raised during setup.
        # For example, 'request_cluster_metrics' to specify custom metrics
        # in the cluster and similarity views.
        attach_plugins(self, plugins=kwargs.get('plugins', None),
                       config_dir=config_dir)

        self._set_cache()
        self.supervisor = self._set_supervisor()
        self.selector = self._set_selector()
        self.color_selector = ColorSelector()

    # Internal methods
    # -------------------------------------------------------------------------

    def _set_cache(self):
        memcached = ('get_template_counts',
                     'get_template_for_cluster',
                     'get_best_channel',
                     'get_best_channels',
                     'get_probe_depth',
                     )
        cached = ('_get_waveforms',
                  '_get_template_waveforms',
                  '_get_features',
                  '_get_template_features',
                  '_get_amplitudes',
                  '_get_correlograms',
                  )
        _cache_methods(self, memcached, cached)

    def _set_supervisor(self):
        # Load the new cluster id.
        new_cluster_id = self.context.load('new_cluster_id'). \
            get('new_cluster_id', None)
        cluster_groups = self.model.get_metadata('group')
        # Special cluster metrics.
        cluster_metrics = {
            'channel': self.get_best_channel,
            'depth': self.get_probe_depth,
            'amplitude': self.get_cluster_amplitude,
        }
        supervisor = Supervisor(spike_clusters=self.model.spike_clusters,
                                cluster_groups=cluster_groups,
                                cluster_metrics=cluster_metrics,
                                cluster_labels=self.model.metadata,
                                quality=self.get_cluster_amplitude,
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
            @supervisor.actions.add(shortcut='shift+ctrl+k')
            def split_init(cluster_ids=None):
                """Split a cluster according to the original templates."""
                if cluster_ids is None:
                    cluster_ids = supervisor.selected
                s = supervisor.clustering.spikes_in_clusters(cluster_ids)
                supervisor.actions.split(s, self.model.spike_templates[s])

        # Save.
        @connect(sender=supervisor)
        def on_request_save(sender, spike_clusters, groups, *labels):
            """Save the modified data."""
            # Save the clusters.
            self.model.save_spike_clusters(spike_clusters)
            # Save cluster metadata.
            for name, values in labels:
                self.model.save_metadata(name, values)
            # Save mean waveforms.
            cluster_ids = self.supervisor.clustering.cluster_ids
            mean_waveforms = {cluster_id: self._get_mean_waveforms(cluster_id)
                              for cluster_id in cluster_ids}
            self.model.save_mean_waveforms(mean_waveforms)

        return supervisor

    def _set_selector(self):
        def spikes_per_cluster(cluster_id):
            return self.supervisor.clustering.spikes_per_cluster.get(cluster_id, [0])
        return Selector(spikes_per_cluster)

    # Model methods
    # -------------------------------------------------------------------------

    def get_template_counts(self, cluster_id):
        """Return a histogram of the number of spikes in each template for
        a given cluster."""
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

    def get_cluster_amplitude(self, cluster_id):
        bunch = self._get_template_waveforms(cluster_id)
        data = bunch.data
        masks = bunch.masks
        assert data.ndim == 3
        n_templates, n_samples, n_channels = data.shape
        assert masks.shape == (n_templates, n_channels)
        template_amplitudes = (data.max(axis=1) - data.min(axis=1)).max(axis=1)
        assert template_amplitudes.shape == (n_templates,)
        return (template_amplitudes * masks[:, 0]).sum()

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

    def get_best_channel(self, cluster_id):
        """Return the best channel of a given cluster."""
        template_id = self.get_template_for_cluster(cluster_id)
        return self.model.get_template(template_id).best_channel

    def get_best_channels(self, cluster_id):
        """Return the best channels of a given cluster."""
        template_id = self.get_template_for_cluster(cluster_id)
        return self.model.get_template(template_id).channel_ids

    def get_probe_depth(self, cluster_id):
        """Return the depth of a cluster."""
        channel_id = self.get_best_channel(cluster_id)
        return self.model.channel_positions[channel_id][1]

    # Waveforms
    # -------------------------------------------------------------------------

    def _get_waveforms(self, cluster_id):
        """Return a selection of waveforms for a cluster."""
        pos = self.model.channel_positions
        spike_ids = self.selector.select_spikes([cluster_id],
                                                self.n_spikes_waveforms,
                                                self.batch_size_waveforms,
                                                )
        channel_ids = self.get_best_channels(cluster_id)
        data = self.model.get_waveforms(spike_ids, channel_ids)
        data = data - data.mean() if data is not None else None
        return Bunch(data=data,
                     channel_ids=channel_ids,
                     channel_positions=pos[channel_ids],
                     )

    def _get_mean_waveforms(self, cluster_id):
        b = self._get_waveforms(cluster_id)
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
        # Get masks.
        masks = count / float(count.max())
        masks = np.tile(masks.reshape((-1, 1)), (1, len(channel_ids)))
        # Get the mean amplitude for the cluster.
        mean_amp = self._get_amplitudes(cluster_id).y.mean()
        # Get all templates from which this cluster stems from.
        templates = [self.model.get_template(template_id)
                     for template_id in template_ids]
        # Construct the waveforms array.
        ns = self.model.n_samples_templates
        data = np.zeros((len(template_ids), ns, self.model.n_channels))
        for i, b in enumerate(templates):
            data[i][:, b.channel_ids] = b.template * mean_amp
        waveforms = data[..., channel_ids]
        assert waveforms.shape == (len(template_ids), ns, len(channel_ids))
        return Bunch(data=waveforms,
                     channel_ids=channel_ids,
                     channel_positions=pos[channel_ids],
                     masks=masks,
                     alpha=1.,
                     )

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
                # Both checkboxes are mutually exclusive.
                if checked:
                    v.actions.get('toggle_templates').setChecked(False)
                v.show_what = 'mean_waveforms' if checked else 'waveforms'
                v.waveforms = funs[v.show_what]
                v.on_select(cluster_ids=v.cluster_ids)

            v.actions.separator()

        return v

    # Features
    # -------------------------------------------------------------------------

    def _get_spike_ids(self, cluster_id=None, load_all=None):
        nsf = self.n_spikes_features
        if cluster_id is None:
            # Background points.
            ns = self.model.n_spikes
            spike_ids = np.arange(0, ns, max(1, ns // nsf))
        else:
            # Load all spikes from the cluster if load_all is True.
            n = nsf if not load_all else None
            spike_ids = self.selector.select_spikes([cluster_id], n)
        # Remove spike_ids that do not belong to model.features_rows
        if self.model.features_rows is not None:  # pragma: no cover
            spike_ids = np.intersect1d(spike_ids, self.model.features_rows)
        return spike_ids

    def _get_spike_times(self, cluster_id=None, load_all=None):
        spike_ids = self._get_spike_ids(cluster_id, load_all=load_all)
        return Bunch(data=self.model.spike_times[spike_ids],
                     spike_ids=spike_ids,
                     lim=(0., self.model.duration))

    def _get_features(self, cluster_id=None, channel_ids=None, load_all=None):
        spike_ids = self._get_spike_ids(cluster_id, load_all=load_all)
        # Use the best channels only if a cluster is specified and
        # channels are not specified.
        if cluster_id is not None and channel_ids is None:
            channel_ids = self.get_best_channels(cluster_id)
        data = self.model.get_features(spike_ids, channel_ids)
        assert data.shape[:2] == (len(spike_ids), len(channel_ids))
        # Remove rows with at least one nan value.
        nan = np.unique(np.nonzero(np.isnan(data))[0])
        nonnan = np.setdiff1d(np.arange(len(spike_ids)), nan)
        data = data[nonnan, ...]
        spike_ids = spike_ids[nonnan]
        assert data.shape[:2] == (len(spike_ids), len(channel_ids))
        assert np.isnan(data).sum() == 0
        return Bunch(data=data,
                     spike_ids=spike_ids,
                     channel_ids=channel_ids,
                     )

    def create_feature_view(self):
        if self.model.features is None:
            return
        return FeatureView(
            features=self._get_features,
            attributes={'time': self._get_spike_times}
        )

    # Template features
    # -------------------------------------------------------------------------

    def _get_template_features(self, cluster_ids):
        assert len(cluster_ids) == 2
        clu0, clu1 = cluster_ids

        s0 = self._get_spike_ids(clu0)
        s1 = self._get_spike_ids(clu1)

        n0 = self.get_template_counts(clu0)
        n1 = self.get_template_counts(clu1)

        t0 = self.model.get_template_features(s0)
        t1 = self.model.get_template_features(s1)

        x0 = np.average(t0, weights=n0, axis=1)
        y0 = np.average(t0, weights=n1, axis=1)

        x1 = np.average(t1, weights=n0, axis=1)
        y1 = np.average(t1, weights=n1, axis=1)

        return Bunch(x0=x0, y0=y0, x1=x1, y1=y1,
                     data_bounds=(min(x0.min(), x1.min()),
                                  min(y0.min(), y1.min()),
                                  max(y0.max(), y1.max()),
                                  max(y0.max(), y1.max()),
                                  ),
                     )

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

        traces_interval = select_traces(m.traces, interval,
                                        sample_rate=m.sample_rate)
        # Reorder vertically.
        out = Bunch(data=traces_interval)
        out.waveforms = []

        def gbc(cluster_id):
            return self.get_best_channels(cluster_id)

        for b in _iter_spike_waveforms(interval=interval,
                                       traces_interval=traces_interval,
                                       model=self.model,
                                       supervisor=self.supervisor,
                                       color_selector=self.color_selector,
                                       n_samples_waveforms=k,
                                       get_best_channels=gbc,
                                       show_all_spikes=show_all_spikes,
                                       ):
            i = b.spike_id
            # Compute the residual: waveform - amplitude * template.
            residual = b.copy()
            template_id = m.spike_templates[i]
            template = m.get_template(template_id).template
            amplitude = m.amplitudes[i]
            residual.data = residual.data - amplitude * template
            out.waveforms.extend([b, residual])
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
        if self.model.traces is None:
            return

        m = self.model
        v = TraceView(traces=self._get_traces,
                      spike_times=self._trace_spike_times,
                      n_channels=m.n_channels,
                      sample_rate=m.sample_rate,
                      duration=m.duration,
                      channel_vertical_order=m.channel_vertical_order,
                      )

        # Update the get_traces() function with show_all_spikes.
        def get_traces(interval):
            return self._get_traces(interval, show_all_spikes=v.show_all_spikes)
        v.traces = get_traces

        @connect(sender=v)
        def on_spike_click(sender, channel_id=None, spike_id=None, cluster_id=None):
            # Select the corresponding cluster.
            self.supervisor.select([cluster_id])
            # Update the trace view.
            v.on_select([cluster_id])

        return v

    # Correlograms
    # -------------------------------------------------------------------------

    def _get_correlograms(self, cluster_ids, bin_size, window_size):
        spike_ids = self.selector.select_spikes(cluster_ids,
                                                self.n_spikes_correlograms,
                                                subset='random',
                                                )
        st = self.model.spike_times[spike_ids]
        sc = self.supervisor.clustering.spike_clusters[spike_ids]
        return correlograms(st,
                            sc,
                            sample_rate=self.model.sample_rate,
                            cluster_ids=cluster_ids,
                            bin_size=bin_size,
                            window_size=window_size,
                            )

    def _get_firing_rate(self, cluster_ids, bin_size):
        spike_ids = self.selector.select_spikes(cluster_ids,
                                                self.n_spikes_correlograms,
                                                subset='random',
                                                )
        sc = self.supervisor.clustering.spike_clusters[spike_ids]
        return firing_rate(
            sc, cluster_ids=cluster_ids, bin_size=bin_size, duration=self.model.duration)

    def create_correlogram_view(self):
        m = self.model
        return CorrelogramView(
            correlograms=self._get_correlograms,
            firing_rate=self._get_firing_rate,
            sample_rate=m.sample_rate,
        )

    # Amplitudes
    # -------------------------------------------------------------------------

    def _get_amplitudes(self, cluster_id):
        n = self.n_spikes_amplitudes
        m = self.model
        spike_ids = self.selector.select_spikes([cluster_id], n)
        x = m.spike_times[spike_ids]
        y = m.amplitudes[spike_ids]
        return Bunch(x=x, y=y, data_bounds=(0., 0., m.duration, y.max()))

    def create_amplitude_view(self):
        if self.model.amplitudes is None:
            return
        return AmplitudeView(coords=self._get_amplitudes)

    # Probe view
    # -------------------------------------------------------------------------

    def create_probe_view(self):
        return ProbeView(
            positions=self.model.channel_positions,
            best_channels=self.get_best_channels,
        )

    # GUI
    # -------------------------------------------------------------------------

    def create_gui(self, **kwargs):
        gui = GUI(name=self.gui_name,
                  subtitle=self.model.dat_path,
                  config_dir=self.config_dir,
                  local_path=op.join(self.cache_dir, 'state.json'),
                  default_state_path=op.join(op.dirname(__file__), 'static/state.json'),
                  view_creator=self.view_creator,
                  view_count={view_cls: 1 for view_cls in self.view_creator.keys()},
                  **kwargs)
        self.supervisor.attach(gui)

        gui.create_views()

        @connect(sender=gui)
        def on_add_view(sender, view):
            view.on_select(cluster_ids=self.supervisor.selected_clusters)

        # Save the memcache when closing the GUI.
        @connect(sender=gui)
        def on_close(sender):
            unconnect(on_add_view)
            # Gather all GUI state attributes from views that are local and thus need
            # to be saved in the data directory.
            gui.state._local_keys = set().union(
                *(getattr(view, 'local_state_attrs', ()) for view in gui.views))
            self.context.save_memcache()

        emit('gui_ready', self, gui)

        return gui


#------------------------------------------------------------------------------
# Template commands
#------------------------------------------------------------------------------

def template_gui(params_path):  # pragma: no cover
    # Create a `phy.log` log file with DEBUG level.
    _add_log_file(op.join(op.dirname(params_path), 'phy.log'))

    params = _read_python(params_path)
    if not os.path.isabs(params['dat_path']):
        params['dat_path'] = op.join(op.dirname(params_path), params['dat_path'])
    params['dtype'] = np.dtype(params['dtype'])

    create_app()
    controller = TemplateController(**params)
    gui = controller.create_gui()
    gui.show()
    run_app()
    gui.close()


def template_describe(params_path):
    """Describe a template dataset."""
    params = _read_python(params_path)
    if not os.path.isabs(params['dat_path']):
        params['dat_path'] = op.join(op.dirname(params_path), params['dat_path'])
    TemplateModel(**params).describe()
