"""Manifest-gated template controller for bounded recording evidence."""

from copy import deepcopy
from pathlib import Path

import numpy as np
from phylib.io.model import get_template_params
from phylib.utils import Bunch, connect, unconnect

from phy.cluster._propositions import (
    MergePropositionController,
    decode_curation_mapping,
    decode_review_mapping,
)

from ..base import _allocate_spike_counts, _sample_spikes_evenly
from ..template.gui import TemplateController
from .curation import BoundedSession
from .model import BoundedTemplateModel
from .statistics import window_correlograms, window_isi
from .supervisor import BoundedSupervisor
from .views import EvidenceAmplitudeView, SampledCorrelogramView, SampledISIView


class BoundedTemplateController(TemplateController):
    gui_name = 'BoundedTemplateGUI'
    # Bounded mode has its own state namespace and initial view list. Importing
    # ordinary saved dock geometry would suppress the new evidence views.
    default_state_path = None
    default_views = (
        'UnitStabilityView',
        'WaveformView',
        'EvidenceAmplitudeView',
        'FeatureView',
        'ProbeView',
        'SampledISIView',
        'SampledCorrelogramView',
    )

    def __init__(self, dir_path=None, model=None, **kwargs):
        if model is not None:
            raise ValueError(
                'Open bounded packages from their directory, without a supplied model.'
            )
        self.bounded_session = BoundedSession(dir_path)
        self.evidence_window_index = 0
        kwargs['enable_threading'] = False
        kwargs.update(get_template_params(Path(dir_path) / 'params.py'))
        kwargs.pop('dir_path', None)
        super().__init__(dir_path=dir_path, **kwargs)

    def _create_model(self, dir_path=None, **kwargs):
        model = BoundedTemplateModel(dir_path=dir_path, session=self.bounded_session, **kwargs)
        if model.sample_rate != self.bounded_session.manifest['sample_rate']:
            model.close()
            raise ValueError('Bounded params sampling rate disagrees with the manifest.')
        model.spike_clusters = self.bounded_session.assignments.copy()
        model.metadata = deepcopy(self.bounded_session.metadata)
        model.duration = self.bounded_session.manifest['num_samples'] / model.sample_rate

        def reject_save(*args, **kwargs):
            raise ValueError('Save bounded curation through supervisor.save() as one transaction.')

        model.save_spike_clusters = reject_save
        model.save_metadata = reject_save
        model.save_spikes_subset_waveforms = reject_save
        model.freeze_source()
        return model

    def _cache_methods(self):
        # Ordinary persistent feature caches omit the current evidence window.
        # Bounded queries are already capped; evaluate them against live state.
        return

    def close(self, close_model=True):
        super().close(close_model=close_model)
        if close_model:
            self.bounded_session.close()

    def _set_supervisor(self):
        session = self.bounded_session
        propositions = self._load_merge_propositions()
        self.supervisor = BoundedSupervisor(
            spike_clusters=self.model.spike_clusters.copy(),
            cluster_groups=self.model.metadata.get('group', {}),
            cluster_metrics=self.cluster_metrics,
            cluster_labels=self.model.metadata,
            similarity=self.similarity_functions[self.similarity],
            session=session,
            merge_propositions=propositions,
        )
        self.supervisor._validate_native()

    def _load_merge_propositions(self):
        session = self.bounded_session
        session.validate_proposition_documents()
        self._merge_proposition_sha256 = None
        if not self.enable_merge_propositions:
            return None
        item = session.proposition_documents.get('curation.json')
        if item is None:
            return None
        from .curation import _read_json

        catalog = decode_curation_mapping(
            _read_json(session.path / item['filename']), retired_units=session.retired_units
        )
        self._merge_proposition_sha256 = item['sha256']
        reviews = session.extra_documents.get('curation_review.json')
        if reviews is not None:
            _, mapping = decode_review_mapping(reviews)
            catalog = catalog._replace(reviews=mapping)
        return MergePropositionController(catalog)

    def _set_cluster_metrics(self):
        super()._set_cluster_metrics()
        self.cluster_metrics['full_count'] = self.get_full_count
        self.cluster_metrics['evidence_count'] = lambda c: self.supervisor.n_spikes(c)
        self.cluster_metrics['waveform_count'] = self.get_waveform_count
        self.cluster_metrics['evidence'] = lambda c: (
            'pending replay' if self.get_full_count(c) is None else 'sampled windows'
        )

    def _set_view_creator(self):
        super()._set_view_creator()
        for name in (
            'ISIView',
            'CorrelogramView',
            'FiringRateView',
            'TraceView',
            'TraceImageView',
        ):
            self.view_creator.pop(name, None)
        self.view_creator.pop('AmplitudeView', None)
        self.view_creator['EvidenceAmplitudeView'] = self.create_evidence_amplitude_view
        self.view_creator['UnitStabilityView'] = self.create_unit_stability_view
        self.view_creator['SampledISIView'] = lambda: SampledISIView(cluster_stat=self._get_isi)
        self.view_creator['SampledCorrelogramView'] = lambda: SampledCorrelogramView(
            correlograms=self._window_correlograms, sample_rate=self.model.sample_rate
        )

    def _get_waveforms_dict(self):
        functions = super()._get_waveforms_dict()
        if not self.bounded_session.manifest['capabilities']['stored_waveforms']:
            functions.pop('waveforms', None)
            functions.pop('mean_waveforms', None)
        return functions

    def create_waveform_view(self):
        view = super().create_waveform_view()
        if view is None:
            return
        plot = view.plot

        def plot_and_clear_preview(**kwargs):
            # Sampling, waveform mode, and channel layout may change the drawn
            # traces. Require a fresh preview rather than leaving stale colors.
            result = plot(**kwargs)
            for amplitude_view in getattr(self, '_evidence_amplitude_views', ()):
                amplitude_view.clear_amplitude_split_threshold()
            return result

        view.plot = plot_and_clear_preview
        return view

    def create_feature_view(self):
        if (
            self.model.features is None
            and not self.bounded_session.manifest['capabilities']['stored_waveforms']
        ):
            return
        view = super().create_feature_view()
        if view is not None:
            view.ex_status = self._window_status()
        return view

    def _source_units(self, cluster_id):
        version = self.bounded_session.events
        cached = getattr(self, '_lineage_cache', None)
        if cached is not None and cached[0] is version:
            return cached[1].get(int(cluster_id))
        units = {int(c): {int(c)} for c in self.bounded_session.unit_ids}
        for event in self.bounded_session.events:
            if event['operation'] == 'merge':
                parents = [units.pop(c) for c in event['parents']]
                units[event['target']] = (
                    None if any(p is None for p in parents) else set.union(*parents)
                )
            elif event['operation'] == 'threshold_split':
                units.pop(event['parent'])
                units.update(dict.fromkeys(event['children']))
        self._lineage_cache = version, units
        return units.get(int(cluster_id))

    def get_full_count(self, cluster_id):
        units = self._source_units(cluster_id)
        if units is None:
            return None
        if not hasattr(self, '_full_counts'):
            self._full_counts = dict(
                zip(self.bounded_session.unit_ids, self.bounded_session.manifest['counts'])
            )
        counts = self._full_counts
        return sum(int(counts[c]) for c in units)

    def get_waveform_count(self, cluster_id):
        stored = self.model.spike_waveforms
        if stored is None:
            return 0
        if (
            not hasattr(self, '_waveform_membership_cache')
            or self._waveform_membership_cache[0] is not self.bounded_session.events
        ):
            labels = self.supervisor.clustering.spike_clusters[stored.spike_ids]
            ids, counts = np.unique(labels, return_counts=True)
            self._waveform_membership_cache = self.bounded_session.events, dict(zip(ids, counts))
        return int(self._waveform_membership_cache[1].get(cluster_id, 0))

    def get_mean_firing_rate(self, cluster_id):
        count = self.get_full_count(cluster_id)
        if count is None:
            return None
        intervals = np.asarray(self.bounded_session.manifest['valid_intervals'])
        exposure = np.sum(intervals[:, 1] - intervals[:, 0]) / self.model.sample_rate
        return count / exposure

    @property
    def evidence_windows(self):
        if not hasattr(self, '_evidence_windows'):
            self._evidence_windows = np.load(
                self.dir_path / self.bounded_session.manifest['windows_file']
            )
        return self._evidence_windows

    def _get_isi(self, cluster_id):
        ids = self.supervisor.clustering.spikes_per_cluster.get(cluster_id, [])
        samples = self.model.spike_samples[ids]
        return Bunch(data=window_isi(samples, self.evidence_windows, self.model.sample_rate))

    def _window_correlograms(self, cluster_ids, bin_size, window_size):
        return window_correlograms(
            self.model.spike_samples,
            self.supervisor.clustering.spike_clusters,
            self.evidence_windows,
            cluster_ids,
            self.model.sample_rate,
            bin_size,
            window_size,
        )

    def create_unit_stability_view(self):
        from .stability import UnitStabilityView

        return UnitStabilityView(self)

    @property
    def current_evidence_window(self):
        return self.evidence_windows[self.evidence_window_index]

    def _window_ids(self, cluster_id=None):
        start, end = self.current_evidence_window
        if cluster_id is None:
            first, last = np.searchsorted(self.model.spike_samples, [start, end])
            return np.arange(first, last, dtype=np.int64)
        ids = self.supervisor.clustering.spikes_per_cluster.get(
            cluster_id, np.array([], dtype=np.int64)
        )
        times = self.model.spike_samples[ids]
        first, last = np.searchsorted(times, [start, end])
        return ids[first:last]

    def _window_time(self, ids):
        # Subtract int64 samples before floating conversion, even at hour 1000.
        return (
            self.model.spike_samples[ids] - self.current_evidence_window[0]
        ) / self.model.sample_rate

    def _get_feature_view_spike_ids(self, cluster_id=None, load_all=False):
        ids = self._window_ids(cluster_id)
        if self.model.features is None:
            if self.model.spike_waveforms is None:
                return np.array([], dtype=np.int64)
            ids = np.intersect1d(ids, self.model.spike_waveforms.spike_ids)
        if getattr(self.model, 'features_rows', None) is not None:
            ids = np.intersect1d(ids, self.model.features_rows)
        limit = self.n_spikes_features_background if cluster_id is None else self.n_spikes_features
        return ids if load_all else _sample_spikes_evenly(ids, limit)

    def _get_feature_view_spike_times(self, cluster_id=None, load_all=False):
        ids = self._get_feature_view_spike_ids(cluster_id, load_all=load_all)
        if not len(ids):
            return
        start, end = self.current_evidence_window
        return Bunch(
            data=self._window_time(ids),
            spike_ids=ids,
            lim=(0.0, (end - start) / self.model.sample_rate),
        )

    def _resolve_spike_amplitudes(self, spike_ids, name, first_cluster):
        if name == 'source':
            return self.bounded_session.attributes['amplitudes.npy'].reshape(-1)[spike_ids]
        return super()._resolve_spike_amplitudes(spike_ids, name, first_cluster)

    def _evidence_amplitudes(self, cluster_ids, load_all=False, scope='window'):
        def eligible_ids(cluster_id=None):
            if cluster_id is None:
                # Sample conceptual row coordinates before gathering assignments.
                # Interactive updates must not scan all retained spikes.
                first, last = 0, self.model.n_spikes
                if scope == 'window':
                    first, last = np.searchsorted(
                        self.model.spike_samples, self.current_evidence_window
                    )
                count = min(last - first, self.n_spikes_amplitudes_background)
                if not count:
                    return np.array([], dtype=np.int64)
                ids = np.arange(count, dtype=np.int64)
                if count > 1:
                    ids *= last - first - 1
                    ids //= count - 1
                return ids + first
            if scope == 'window':
                return self._window_ids(cluster_id)
            return self.supervisor.clustering.spikes_per_cluster.get(
                cluster_id, np.array([], dtype=np.int64)
            )

        def spike_times(ids):
            if scope == 'window':
                return self._window_time(ids)
            return self.model.spike_samples[ids] / self.model.sample_rate

        active = [c for c in cluster_ids if c is not None]
        eligible = [eligible_ids(c) for c in active]
        counts = _allocate_spike_counts(
            [len(ids) for ids in eligible],
            per_cluster=None if load_all else self.n_spikes_amplitudes,
            total=None if load_all else self.n_spikes_amplitudes_total,
        )
        values = self.bounded_session.attributes['amplitudes.npy'].reshape(-1)
        output = []
        if None in cluster_ids:
            ids = eligible_ids()
            ids = ids[~np.isin(self.supervisor.clustering.spike_clusters[ids], active)]
            ids = _sample_spikes_evenly(ids, self.n_spikes_amplitudes_background)
            ids = ids[np.isfinite(values[ids])]
            output.append(
                Bunch(amplitudes=values[ids], spike_ids=ids, spike_times=spike_times(ids))
            )
        for ids, count in zip(eligible, counts):
            ids = _sample_spikes_evenly(ids, int(count))
            ids = ids[np.isfinite(values[ids])]
            output.append(
                Bunch(amplitudes=values[ids], spike_ids=ids, spike_times=spike_times(ids))
            )
        return output

    def create_evidence_amplitude_view(self):
        array = self.bounded_session.attributes.get('amplitudes.npy')
        if array is None or (array.ndim != 1 and array.shape[1:] != (1,)):
            return
        start, end = self.current_evidence_window

        def amplitudes(cluster_ids, load_all=False):
            return self._evidence_amplitudes(
                cluster_ids, load_all=load_all, scope=view.amplitude_scope
            )

        def split_is_eligible():
            return (
                len(self.supervisor.selected) == 1
                and not self.supervisor.selection.state.is_merge_mode
                and 'amplitudes.npy'
                in self.bounded_session.capabilities['predicate_split_attributes']
            )

        view = EvidenceAmplitudeView(
            amplitudes={'source': amplitudes},
            amplitudes_type='source',
            duration=(end - start) / self.model.sample_rate,
            split_is_eligible=split_is_eligible,
        )
        view._controller = self
        self._evidence_amplitude_views = getattr(self, '_evidence_amplitude_views', [])
        self._evidence_amplitude_views.append(view)
        self._update_amplitude_scope(view)

        @connect(sender=self.supervisor)
        def on_select(sender, cluster_ids, **kwargs):
            view.clear_amplitude_split_threshold()
            self.supervisor._update_graphical_split_action()

        @connect(sender=self.supervisor)
        def on_bounded_state_changed(sender):
            view.clear_amplitude_split_threshold()

        @connect(event='amplitude_split_preview_changed', sender=view)
        def on_preview(sender, **kwargs):
            self.supervisor._update_graphical_split_action()

        @connect(event='view_attached', sender=view)
        def on_attached(sender, gui):
            def visibility_changed(visible):
                if not visible:
                    view.clear_amplitude_split_threshold()
                self.supervisor._update_graphical_split_action()

            view.dock.visibilityChanged.connect(visibility_changed)

        @connect(event='selected_channel_changed')
        @connect(event='selected_feature_changed')
        def on_context_changed(sender):
            if getattr(sender, 'gui', None) is self.supervisor.gui:
                view.clear_amplitude_split_threshold()

        @connect(event='close_view', sender=view)
        def on_close(view_, gui):
            view.clear_amplitude_split_threshold()
            self._evidence_amplitude_views.remove(view)
            for callback in (
                on_select,
                on_bounded_state_changed,
                on_preview,
                on_context_changed,
                on_attached,
                on_close,
            ):
                unconnect(callback)
            self.supervisor._update_graphical_split_action()

        return view

    def _update_amplitude_scope(self, view):
        start, end = self.current_evidence_window
        overview = view.amplitude_scope == 'overview'
        view.duration = self.model.duration if overview else (end - start) / self.model.sample_rate
        if overview:
            view.window_caption = 'Sampled overview | All retained evidence windows'
            if hasattr(view, 'window_label'):
                view.window_label.setText(view.window_caption)
        else:
            view.set_window_context(
                self.evidence_window_index,
                start / self.model.sample_rate,
                end / self.model.sample_rate,
            )
        view.ex_status = (
            'All retained windows; original recording clock; sampled source amplitudes.'
            if overview
            else self._window_status()
        )
        if hasattr(view, 'time_axis_label'):
            view.time_axis_label.setText(
                'Recording time (seconds)' if overview else 'Seconds from window start'
            )
        view.canvas.panzoom.reset()
        if view.cluster_ids:
            view.plot()
        view.update_status()

    def _window_status(self):
        start, end = self.current_evidence_window
        return (
            f'Evidence window {self.evidence_window_index}: {start / self.model.sample_rate:.3f}–'
            f'{end / self.model.sample_rate:.3f} s; time axis is seconds from window start; nonfinite amplitudes omitted.'
        )

    def inspect_evidence_window(self, index):
        index = int(index)
        if not 0 <= index < len(self.evidence_windows):
            raise ValueError('Evidence window index is outside the retained plan.')
        self.evidence_window_index = index
        for view in getattr(self, '_evidence_amplitude_views', ()):
            view.amplitude_scope = 'window'
        gui = getattr(self.supervisor, 'gui', None)
        if gui is not None:
            for view in gui.list_views('FeatureView'):
                view.ex_status = self._window_status()
                view.canvas.panzoom.reset()
                if view.cluster_ids:
                    view.plot()
                view.update_status()

    def create_gui(self, *args, **kwargs):
        gui = super().create_gui(*args, **kwargs)
        gui.setWindowTitle(
            f'phy — Bounded evidence ({self.model.n_spikes:,} / '
            f'{self.bounded_session.manifest["num_spikes"]:,} spikes)'
        )

        @self.supervisor.actions.add(prompt=True, name='Split by source amplitude')
        def split_by_source_amplitude(threshold):
            """Split one selected unit below a source amplitude threshold (source units)."""
            ids = self.supervisor.selected
            if len(ids) != 1:
                raise ValueError('Select one unit for a source-amplitude threshold split.')
            return self.supervisor.split_source_amplitude(ids[0], float(threshold))

        @gui.view_actions.add(prompt=True, prompt_default=lambda: self.evidence_window_index)
        def inspect_evidence_window(index):
            """Inspect a retained evidence window by its zero-based index."""
            self.inspect_evidence_window(index)

        split_action = self.supervisor.actions.get('split')
        if split_action is not None:
            self.supervisor._update_graphical_split_action()
            split_action.setToolTip(
                'Selected-row splits cannot assign unseen source spikes; use a source-amplitude predicate.'
            )
        predicate_action = self.supervisor.actions.get('Split by source amplitude')
        if predicate_action is not None:
            predicate_action.setEnabled(
                'amplitudes.npy' in self.bounded_session.capabilities['predicate_split_attributes']
            )

        return gui
