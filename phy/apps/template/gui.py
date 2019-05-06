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
from phylib.stats import correlograms, firing_rate
from phylib.utils import Bunch, emit, connect
from phylib.utils._color import ColorSelector
from phylib.utils._misc import _read_python
from phy.cluster.supervisor import Supervisor
from phy.cluster.views import (WaveformView,
                               FeatureView,
                               TraceView as _TraceView,
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

from .model import TemplateModel, from_sparse

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Utils and views
#------------------------------------------------------------------------------

class TraceView(_TraceView):
    show_all_spikes = False

    @property
    def state(self):
        state = super(TraceView, self).state
        state.update(show_all_spikes=self.show_all_spikes)
        return state


class TemplateFeatureView(ScatterView):
    _callback_delay = 100

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

    n_spikes_features = 10000
    n_spikes_amplitudes = 10000
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
        cluster_metrics = {
            'channel': self.get_best_channel,
            'depth': self.get_probe_depth,
        }
        supervisor = Supervisor(spike_clusters=self.model.spike_clusters,
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

    def _add_view(self, gui, view):
        view.attach(gui)
        emit('add_view', self, gui, view)
        return view

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
        data = np.stack([b.template * mean_amp for b in templates], axis=0)
        cols = np.stack([b.channel_ids for b in templates], axis=0)
        # NOTE: transposition because the channels should be in the second
        # dimension for from_sparse.
        data = data.transpose((0, 2, 1))
        assert data.ndim == 3
        assert data.shape[1] == cols.shape[1]
        waveforms = from_sparse(data, cols, channel_ids)
        # Transpose back.
        waveforms = waveforms.transpose((0, 2, 1))
        return Bunch(data=waveforms,
                     channel_ids=channel_ids,
                     channel_positions=pos[channel_ids],
                     masks=masks,
                     alpha=1.,
                     )

    def add_waveform_view(self, gui):
        f = (self._get_waveforms if self.model.traces is not None
             else self._get_template_waveforms)
        v = WaveformView(waveforms=f,
                         )
        v = self._add_view(gui, v)

        v.actions.separator()

        @v.actions.add(shortcut='w', checkable=True)
        def toggle_templates(checked):
            f, g = self._get_waveforms, self._get_template_waveforms
            if self.model.traces is None:
                return
            v.waveforms = f if v.waveforms == g else g
            v.on_select(cluster_ids=v.cluster_ids)

        @v.actions.add(shortcut='m', checkable=True)
        def toggle_mean_waveforms(checked):
            f, g = self._get_waveforms, self._get_mean_waveforms
            v.waveforms = f if v.waveforms == g else g
            v.on_select(cluster_ids=v.cluster_ids)

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
        if self.model.features_rows is not None:
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

    def add_feature_view(self, gui):
        v = FeatureView(features=self._get_features,
                        attributes={'time': self._get_spike_times}
                        )
        return self._add_view(gui, v)

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

    def add_template_feature_view(self, gui):
        v = TemplateFeatureView(coords=self._get_template_features,
                                )
        return self._add_view(gui, v)

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

    def _jump_to_spike(self, view, delta=+1):
        """Jump to next or previous spike from the selected clusters."""
        m = self.model
        cluster_ids = self.supervisor.selected
        if len(cluster_ids) == 0:
            return
        spc = self.supervisor.clustering.spikes_per_cluster
        spike_ids = spc[cluster_ids[0]]
        spike_times = m.spike_times[spike_ids]
        ind = np.searchsorted(spike_times, view.time)
        n = len(spike_times)
        view.go_to(spike_times[(ind + delta) % n])

    def add_trace_view(self, gui):
        m = self.model
        v = TraceView(traces=self._get_traces,
                      n_channels=m.n_channels,
                      sample_rate=m.sample_rate,
                      duration=m.duration,
                      channel_vertical_order=m.channel_vertical_order,
                      )
        self._add_view(gui, v)

        # Update the get_traces() function with show_all_spikes.
        def get_traces(interval):
            return self._get_traces(interval, show_all_spikes=v.show_all_spikes)
        v.traces = get_traces

        v.actions.separator()

        @v.actions.add(shortcut='alt+pgdown')
        def go_to_next_spike():
            """Jump to the next spike from the first selected cluster."""
            self._jump_to_spike(v, +1)

        @v.actions.add(shortcut='alt+pgup')
        def go_to_previous_spike():
            """Jump to the previous spike from the first selected cluster."""
            self._jump_to_spike(v, -1)

        v.actions.separator()

        @v.actions.add(shortcut='alt+s', checkable=True, checked=v.show_all_spikes)
        def toggle_highlighted_spikes(checked):
            """Toggle between showing all spikes or selected spikes."""
            v.show_all_spikes = checked
            v.set_interval()

        @connect
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

    def add_correlogram_view(self, gui):
        m = self.model
        v = CorrelogramView(correlograms=self._get_correlograms,
                            firing_rate=self._get_firing_rate,
                            sample_rate=m.sample_rate,
                            )
        return self._add_view(gui, v)

    # Amplitudes
    # -------------------------------------------------------------------------

    def _get_amplitudes(self, cluster_id):
        n = self.n_spikes_amplitudes
        m = self.model
        spike_ids = self.selector.select_spikes([cluster_id], n)
        x = m.spike_times[spike_ids]
        y = m.amplitudes[spike_ids]
        return Bunch(x=x, y=y, data_bounds=(0., 0., m.duration, y.max()))

    def add_amplitude_view(self, gui):
        v = AmplitudeView(coords=self._get_amplitudes,
                          )
        return self._add_view(gui, v)

    # Probe view
    # -------------------------------------------------------------------------

    def add_probe_view(self, gui):
        v = ProbeView(positions=self.model.channel_positions,
                      best_channels=self.get_best_channels,
                      )
        v.attach(gui)
        return v

    # GUI
    # -------------------------------------------------------------------------

    def create_gui(self, **kwargs):
        gui = GUI(name=self.gui_name,
                  subtitle=self.model.dat_path,
                  config_dir=self.config_dir,
                  **kwargs)

        self.supervisor.attach(gui)

        self.add_waveform_view(gui)

        if self.model.traces is not None:
            self.add_trace_view(gui)
        else:
            logger.warning(
                "The raw data file is not available, the trace view won't be displayed.")

        if self.model.features is not None:
            self.add_feature_view(gui)
        else:
            logger.warning(
                "Features file is not available, the feature view won't be displayed.")

        if self.model.template_features is not None:
            self.add_template_feature_view(gui)
        else:
            logger.warning(
                "Template feature file is not available, "
                "the template feature view won't be displayed.")

        self.add_correlogram_view(gui)

        if self.model.amplitudes is not None:
            self.add_amplitude_view(gui)
        else:
            logger.warning(
                "The amplitude file is not available, the amplitude view won't be displayed.")

        self.add_probe_view(gui)

        # Save the memcache when closing the GUI.
        @connect(sender=gui)
        def on_close(sender):
            self.context.save_memcache()

        emit('gui_ready', self, gui)

        return gui


#------------------------------------------------------------------------------
# Template commands
#------------------------------------------------------------------------------

def template_gui(params_path):
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
