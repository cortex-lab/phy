# -*- coding: utf-8 -*-

"""Kwik GUI."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
from operator import itemgetter
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np

from phylib.stats import correlograms
from phylib.stats.clusters import get_waveform_amplitude
from phylib.io.array import Selector
from phylib.utils import Bunch, emit, connect, unconnect
from phylib.utils._color import ClusterColorSelector
from phy.cluster.supervisor import Supervisor
from phy.cluster.views import (WaveformView,
                               FeatureView,
                               TraceView,
                               CorrelogramView,
                               select_traces,
                               )
from phy.cluster.views.trace import _iter_spike_waveforms
from phy.gui import create_app, run_app, GUI
from phy.utils.context import Context, _cache_methods
from phy.utils.plugin import attach_plugins
from .. import _add_log_file


logger = logging.getLogger(__name__)

try:
    from klusta.kwik import KwikModel
    from klusta.launch import cluster
except ImportError:  # pragma: no cover
    logger.warning("Package klusta not installed: the KwikGUI will not work.")


#------------------------------------------------------------------------------
# Kwik GUI
#------------------------------------------------------------------------------

def _backup(path):
    """Backup a file."""
    path_backup = str(path) + '.bak'
    if not Path(path_backup).exists():
        logger.info("Backup `%s`.", path_backup)
        shutil.copy(path, path_backup)


def _get_distance_max(pos):
    return np.sqrt(np.sum(pos.max(axis=0) - pos.min(axis=0)) ** 2)


class KwikController(object):
    gui_name = 'KwikGUI'

    n_spikes_waveforms = 100
    batch_size_waveforms = 10

    n_spikes_features = 10000
    n_spikes_amplitudes = 10000

    n_spikes_close_clusters = 100
    n_closest_channels = 16

    def __init__(self, kwik_path, config_dir=None, **kwargs):
        kwik_path = Path(kwik_path)
        _backup(kwik_path)
        self.model = KwikModel(str(kwik_path), **kwargs)
        m = self.model
        self.channel_vertical_order = np.argsort(m.channel_positions[:, 1])
        self.distance_max = _get_distance_max(self.model.channel_positions)
        self.cache_dir = kwik_path.parent / '.phy'
        cg = kwargs.get('channel_group', None)
        if cg is not None:
            self.cache_dir = self.cache_dir / str(cg)
        self.context = Context(self.cache_dir)
        self.config_dir = config_dir
        self.view_creator = {
            WaveformView: self.create_waveform_view,
            TraceView: self.create_trace_view,
            FeatureView: self.create_feature_view,
            CorrelogramView: self.create_correlogram_view,
        }

        self._set_cache()
        self.supervisor = self._set_supervisor()
        self.selector = self._set_selector()
        self.color_selector = ClusterColorSelector(
            cluster_labels=self.supervisor.cluster_labels,
            cluster_metrics=self.supervisor.cluster_metrics,
            cluster_ids=self.supervisor.clustering.cluster_ids,
        )

        attach_plugins(self, plugins=kwargs.get('plugins', None),
                       config_dir=config_dir)

    # Internal methods
    # -------------------------------------------------------------------------

    def _set_cache(self):
        memcached = ('get_best_channels',
                     'get_probe_depth',
                     '_get_mean_masks',
                     '_get_mean_waveforms',
                     )
        cached = ('_get_waveforms',
                  '_get_features',
                  '_get_masks',
                  )
        _cache_methods(self, memcached, cached)

    def _set_supervisor(self):
        # Load the new cluster id.
        new_cluster_id = self.context.load('new_cluster_id'). \
            get('new_cluster_id', None)
        cluster_groups = self.model.cluster_groups
        cluster_metrics = {
            'channel': self.get_best_channel,
            'depth': self.get_probe_depth,
        }
        supervisor = Supervisor(self.model.spike_clusters,
                                similarity=self.similarity,
                                cluster_groups=cluster_groups,
                                cluster_metrics=cluster_metrics,
                                new_cluster_id=new_cluster_id,
                                context=self.context,
                                )

        @connect(sender=supervisor)
        def on_attach_gui(sender):
            @supervisor.actions.add
            def recluster():
                """Relaunch KlustaKwik on the selected clusters."""
                # Selected clusters.
                cluster_ids = supervisor.selected
                spike_ids = self.selector.select_spikes(cluster_ids)
                logger.info("Running KlustaKwik on %d spikes.", len(spike_ids))

                # Run KK2 in a temporary directory to avoid side effects.
                n = 10
                with TemporaryDirectory() as tempdir:
                    spike_clusters, metadata = cluster(
                        self.model,
                        spike_ids,
                        num_starting_clusters=n,
                        tempdir=tempdir,
                    )
                self.supervisor.split(spike_ids, spike_clusters)

        # Save.
        @connect(sender=supervisor)
        def on_request_save(sender, spike_clusters, groups, *labels):
            """Save the modified data."""
            groups = {c: g.title() for c, g in groups.items()}
            self.model.save(spike_clusters, groups)

        return supervisor

    def _set_selector(self):
        def spikes_per_cluster(cluster_id):
            return self.supervisor.clustering.spikes_per_cluster[cluster_id]
        return Selector(spikes_per_cluster)

    # Model methods
    # -------------------------------------------------------------------------

    def get_best_channel(self, cluster_id):
        return self.get_best_channels(cluster_id)[0]

    def get_best_channels(self, cluster_id):
        """Only used in the trace view."""
        mm = self._get_mean_masks(cluster_id)
        channel_ids = np.argsort(mm)[::-1]
        ind = mm[channel_ids] > .1
        if np.sum(ind) > 0:
            channel_ids = channel_ids[ind]
        else:  # pragma: no cover
            channel_ids = channel_ids[:4]
        return channel_ids

    def get_cluster_position(self, cluster_id):
        channel_id = self.get_best_channel(cluster_id)
        return self.model.channel_positions[channel_id]

    def get_probe_depth(self, cluster_id):
        return self.get_cluster_position(cluster_id)[1]

    def similarity(self, cluster_id):
        """Return the list of similar clusters to a given cluster."""

        pos_i = self.get_cluster_position(cluster_id)
        assert len(pos_i) == 2

        def _sim_ij(cj):
            """Distance between channel position of clusters i and j."""
            pos_j = self.get_cluster_position(cj)
            assert len(pos_j) == 2
            d = np.sqrt(np.sum((pos_j - pos_i) ** 2))
            return self.distance_max - d

        out = [(cj, _sim_ij(cj))
               for cj in self.supervisor.clustering.cluster_ids]
        return sorted(out, key=itemgetter(1), reverse=True)

    # Waveforms
    # -------------------------------------------------------------------------

    def _get_masks(self, cluster_id):
        spike_ids = self.selector.select_spikes([cluster_id],
                                                self.n_spikes_waveforms,
                                                self.batch_size_waveforms,
                                                )
        return self.model.all_masks[spike_ids]

    def _get_mean_masks(self, cluster_id):
        return np.mean(self._get_masks(cluster_id), axis=0)

    def _get_waveforms(self, cluster_id):
        """Return a selection of waveforms for a cluster."""
        pos = self.model.channel_positions
        spike_ids = self.selector.select_spikes([cluster_id],
                                                self.n_spikes_waveforms,
                                                self.batch_size_waveforms,
                                                )
        data = self.model.all_waveforms[spike_ids]
        mm = self._get_mean_masks(cluster_id)
        mw = np.mean(data, axis=0)
        amp = get_waveform_amplitude(mm, mw)
        masks = self._get_masks(cluster_id)
        # Find the best channels.
        channel_ids = np.argsort(amp)[::-1]
        return Bunch(data=data[..., channel_ids],
                     channel_ids=channel_ids,
                     channel_positions=pos[channel_ids],
                     masks=masks[:, channel_ids],
                     )

    def _get_mean_waveforms(self, cluster_id):
        b = self._get_waveforms(cluster_id).copy()
        b.data = np.mean(b.data, axis=0)[np.newaxis, ...]
        b.masks = np.mean(b.masks, axis=0)[np.newaxis, ...] ** .1
        b['alpha'] = 1.
        return b

    def create_waveform_view(self):
        f = self._get_waveforms
        v = WaveformView(waveforms=f)
        v.shortcuts['toggle_mean_waveforms'] = 'm'

        v.state_attrs += ('show_what',)
        funs = {
            'waveforms': self._get_waveforms,
            'mean_waveforms': self._get_mean_waveforms,
        }

        # Add extra actions.
        @connect(sender=v)
        def on_view_actions_created(sender):
            # NOTE: this callback function is called in WaveformView.attach().

            # Initialize show_what if it was not set in the GUI state.
            if not hasattr(v, 'show_what'):
                v.show_what = 'waveforms'
            # Set the waveforms function.
            v.waveforms = funs[v.show_what]

            @v.actions.add(checkable=True, checked=v.show_what == 'mean_waveforms')
            def toggle_mean_waveforms(checked):
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
            return np.arange(0, ns, max(1, ns // nsf))
        else:
            # Load all spikes from the cluster if load_all is True.
            n = nsf if not load_all else None
            return self.selector.select_spikes([cluster_id], n)

    def _get_spike_times(self, cluster_id=None, load_all=None):
        spike_ids = self._get_spike_ids(cluster_id, load_all=load_all)
        return Bunch(data=self.model.spike_times[spike_ids],
                     lim=(0., self.model.duration))

    def _get_features(self, cluster_id=None, channel_ids=None, load_all=None):
        spike_ids = self._get_spike_ids(cluster_id, load_all=load_all)
        # Use the best channels only if a cluster is specified and
        # channels are not specified.
        if cluster_id is not None and channel_ids is None:
            channel_ids = self.get_best_channels(cluster_id)
        f = self.model.all_features[spike_ids][:, channel_ids]
        m = self.model.all_masks[spike_ids][:, channel_ids]
        return Bunch(data=f,
                     masks=m,
                     spike_ids=spike_ids,
                     channel_ids=channel_ids,
                     )

    def create_feature_view(self):
        return FeatureView(
            features=self._get_features,
            attributes={'time': self._get_spike_times}
        )

    # Traces
    # -------------------------------------------------------------------------

    def _get_traces(self, interval):
        """Get traces and spike waveforms."""
        ns = self.model.n_samples_waveforms
        m = self.model
        c = self.channel_vertical_order

        traces_interval = select_traces(m.traces, interval,
                                        sample_rate=m.sample_rate)
        # Reorder vertically.
        traces_interval = traces_interval[:, c]

        def gbc(cluster_id):
            ch = self.get_best_channels(cluster_id)
            return ch

        out = Bunch(data=traces_interval)
        out.waveforms = []
        for b in _iter_spike_waveforms(interval=interval,
                                       traces_interval=traces_interval,
                                       model=self.model,
                                       supervisor=self.supervisor,
                                       color_selector=self.color_selector,
                                       n_samples_waveforms=ns,
                                       get_best_channels=gbc,
                                       ):
            b.channel_labels = m.channel_order[b.channel_ids]
            out.waveforms.append(b)
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
        m = self.model
        v = TraceView(traces=self._get_traces,
                      spike_times=self._trace_spike_times,
                      n_channels=m.n_channels,
                      sample_rate=m.sample_rate,
                      duration=m.duration,
                      channel_vertical_order=self.channel_vertical_order,
                      )

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
        spike_ids = self.selector.select_spikes(cluster_ids, 100000)
        st = self.model.spike_times[spike_ids]
        sc = self.supervisor.clustering.spike_clusters[spike_ids]
        return correlograms(st,
                            sc,
                            sample_rate=self.model.sample_rate,
                            cluster_ids=cluster_ids,
                            bin_size=bin_size,
                            window_size=window_size,
                            )

    def create_correlogram_view(self):
        m = self.model
        return CorrelogramView(
            correlograms=self._get_correlograms,
            sample_rate=m.sample_rate,
        )

    # GUI
    # -------------------------------------------------------------------------

    def create_gui(self, **kwargs):
        gui = GUI(name=self.gui_name,
                  subtitle=self.model.kwik_path,
                  config_dir=self.config_dir,
                  default_state_path=Path(__file__).parent / 'static/state.json',
                  view_creator=self.view_creator,
                  view_count={view_cls: 1 for view_cls in self.view_creator.keys()},
                  **kwargs)
        self.supervisor.attach(gui)

        gui.create_views()

        # Save the memcache when closing the GUI.
        @connect(sender=gui)
        def on_close(e=None):
            self.context.save_memcache()
            # Unconnect all events GUI and supervisor.
            unconnect(gui, self.supervisor, *gui.views)

        emit('gui_ready', self, gui)

        return gui


#------------------------------------------------------------------------------
# Kwik commands
#------------------------------------------------------------------------------

def kwik_gui(path, channel_group=None, clustering=None):  # pragma: no cover
    # Create a `phy.log` log file with 0 level.
    _add_log_file(Path(path).parent / 'phy.log')

    create_app()
    controller = KwikController(
        path, channel_group=channel_group, clustering=clustering)
    gui = controller.create_gui()
    gui.show()
    run_app()
    gui.close()


def kwik_describe(path, channel_group=None, clustering=None):
    """Describe a template dataset."""
    KwikModel(path, channel_group=channel_group, clustering=clustering).describe()
