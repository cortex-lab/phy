"""Template GUI."""


# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

import logging
from operator import itemgetter
from pathlib import Path

import numpy as np
from phylib import _add_log_file
from phylib.io.model import TemplateModel, get_closest_channels, load_model
from phylib.io.traces import MtscompEphysReader
from phylib.utils import Bunch, connect

from phy.cluster.views import ScatterView
from phy.gui import create_app, run_app

from ..base import BaseController, FeatureMixin, TemplateMixin, TraceMixin, WaveformMixin

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Custom views
# ------------------------------------------------------------------------------


class TemplateFeatureView(ScatterView):
    """Scatter view showing the template features."""


# ------------------------------------------------------------------------------
# Template Controller
# ------------------------------------------------------------------------------


class TemplateController(WaveformMixin, FeatureMixin, TemplateMixin, TraceMixin, BaseController):
    """Controller for the Template GUI.

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
    enable_threading : boolean
        Whether to enable threading in the views when selecting clusters.

    """

    gui_name = 'TemplateGUI'
    enable_merge_propositions = True

    # Specific views implemented in this class.
    _new_views = ('TemplateFeatureView',)

    # Classes to load by default, in that order. The view refresh follows the same order
    # when the cluster selection changes.
    default_views = (
        'WaveformView',
        'CorrelogramView',
        'ISIView',
        'FeatureView',
        'AmplitudeView',
        'FiringRateView',
        'TraceView',
        'ProbeView',
        'TemplateFeatureView',
    )

    # Internal methods
    # -------------------------------------------------------------------------

    def _get_waveforms_dict(self):
        waveforms_dict = super()._get_waveforms_dict()
        # Remove waveforms and mean_waveforms if there is no raw data file.
        if self.model.traces is None and self.model.spike_waveforms is None:
            waveforms_dict.pop('waveforms', None)
            waveforms_dict.pop('mean_waveforms', None)
        return waveforms_dict

    def _create_model(self, dir_path=None, **kwargs):
        return TemplateModel(dir_path=dir_path, **kwargs)

    def _set_supervisor(self):
        super()._set_supervisor()

        supervisor = self.supervisor

        @connect(sender=supervisor)
        def on_attach_gui(sender):
            @supervisor.actions.add(shortcut='shift+ctrl+k', set_busy=True)
            def split_init(cluster_ids=None):
                """Split a cluster according to the original templates."""
                if cluster_ids is None:
                    cluster_ids = supervisor.selected
                s = supervisor.clustering.spikes_in_clusters(cluster_ids)
                supervisor.actions.split(s, self.model.spike_templates[s])

    def _set_similarity_functions(self):
        super()._set_similarity_functions()
        self.similarity_functions['template'] = self.template_similarity
        self.similarity = 'template'

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

    def _set_view_creator(self):
        super()._set_view_creator()
        self.view_creator['TemplateFeatureView'] = self.create_template_feature_view

    # Public methods
    # -------------------------------------------------------------------------

    def _get_stored_waveform_channel_amplitudes(self, cluster_id):
        """Estimate cluster channel amplitudes from an optional stored waveform subset."""
        stored = self.model.spike_waveforms
        if stored is None:
            return None

        rows_by_cluster = getattr(self, '_stored_waveform_rows_by_cluster', None)
        if rows_by_cluster is None or cluster_id not in rows_by_cluster:
            subset_clusters = self.supervisor.clustering.spike_clusters[stored.spike_ids]
            order = np.argsort(subset_clusters, kind='stable')
            cluster_ids, starts = np.unique(subset_clusters[order], return_index=True)
            rows_by_cluster = {
                int(current_cluster): rows
                for current_cluster, rows in zip(cluster_ids, np.split(order, starts[1:]))
            }
            self._stored_waveform_rows_by_cluster = rows_by_cluster

        rows = rows_by_cluster.get(cluster_id)
        if rows is None or len(rows) == 0:
            return None
        waveforms = stored.waveforms[rows]
        channel_ids = stored.spike_channels[rows]
        if waveforms.ndim != 3 or channel_ids.shape != waveforms.shape[::2]:
            logger.warning('Stored spike waveforms have inconsistent waveform/channel shapes.')
            return None

        local_amplitudes = np.ptp(waveforms, axis=1)
        valid = (
            (channel_ids >= 0)
            & (channel_ids < self.model.n_channels)
            & np.isfinite(local_amplitudes)
        )
        if not np.any(valid):
            return None
        sums = np.zeros(self.model.n_channels, dtype=np.float64)
        counts = np.zeros(self.model.n_channels, dtype=np.int64)
        np.add.at(sums, channel_ids[valid], local_amplitudes[valid])
        np.add.at(counts, channel_ids[valid], 1)
        populated = np.flatnonzero(counts)
        amplitudes = np.zeros(self.model.n_channels, dtype=np.float64)
        amplitudes[populated] = sums[populated] / counts[populated]

        best_channel = int(populated[np.argmax(amplitudes[populated])])
        closest = get_closest_channels(
            self.model.channel_positions, best_channel, self.model.n_closest_channels
        )
        selected = np.intersect1d(populated, closest)
        if self.model.channel_shanks is not None:
            selected = selected[
                self.model.channel_shanks[selected] == self.model.channel_shanks[best_channel]
            ]
        selected = selected[np.argsort(amplitudes[selected])[::-1]]
        return selected, amplitudes[selected]

    def get_best_channels(self, cluster_id):
        """Return the best channels of a given cluster."""
        template_id = self.get_template_for_cluster(cluster_id)
        template = self.model.get_template(template_id)
        if not template:
            stored = self._get_stored_waveform_channel_amplitudes(cluster_id)
            if stored is not None:
                return stored[0]
            return [0]
        return template.channel_ids

    def get_channel_amplitudes(self, cluster_id):
        """Return the channel amplitudes of the best channels of a given cluster."""
        template_id = self.get_template_for_cluster(cluster_id)
        template = self.model.get_template(template_id, amplitude_threshold=0.5)
        if not template:
            stored = self._get_stored_waveform_channel_amplitudes(cluster_id)
            if stored is not None:
                channel_ids, amplitude = stored
                m, M = amplitude.min(), amplitude.max()
                d = (M - m) if m < M else 1.0
                return channel_ids, (amplitude - m) / d
            return [0], [0.0]
        m, M = template.amplitude.min(), template.amplitude.max()
        d = (M - m) if m < M else 1.0
        return template.channel_ids, (template.amplitude - m) / d

    def template_similarity(self, cluster_id):
        """Return the list of similar clusters to a given cluster."""
        # Templates of the cluster.
        temp_i = np.nonzero(self.get_template_counts(cluster_id))[0]
        # The similarity of the cluster with each template.
        sims = np.max(self.model.similar_templates[temp_i, :], axis=0)

        def _sim_ij(cj):
            temp_j = np.nonzero(self.get_template_counts(cj))[0]
            return float(np.max(sims[temp_j]))

        out = [(cj, _sim_ij(cj)) for cj in self.supervisor.clustering.cluster_ids]
        # NOTE: hard-limit to 100 for performance reasons.
        return sorted(out, key=itemgetter(1), reverse=True)[:100]

    def get_template_amplitude(self, template_id):
        """Return the maximum amplitude of a template's waveforms across all channels."""
        waveforms = self.model.get_template_waveforms(template_id)
        if waveforms is None:  # pragma: no cover
            return 0
        assert waveforms.ndim == 2  # shape: (n_samples, n_channels)
        return (waveforms.max(axis=0) - waveforms.min(axis=0)).max()

    def create_template_feature_view(self):
        if self.model.template_features is None:
            return
        return TemplateFeatureView(coords=self._get_template_features)


# ------------------------------------------------------------------------------
# Template commands
# ------------------------------------------------------------------------------


def template_gui(params_path, **kwargs):  # pragma: no cover
    """Launch the Template GUI."""
    # Create a `phy.log` log file with DEBUG level.
    p = Path(params_path)
    dir_path = p.parent
    _add_log_file(dir_path / 'phy.log')

    model = load_model(params_path)
    # Automatically export spike waveforms when using compressed raw ephys.
    if model.spike_waveforms is None and isinstance(model.traces, MtscompEphysReader):
        # TODO: customizable values below.
        model.save_spikes_subset_waveforms(max_n_spikes_per_template=500, max_n_channels=16)

    create_app()
    controller = TemplateController(model=model, dir_path=dir_path, **kwargs)
    gui = controller.create_gui()
    gui.show()
    run_app()
    gui.close()
    controller.model.close()


def template_describe(params_path):
    """Describe a template dataset."""
    model = load_model(params_path)
    model.describe()
    model.close()
