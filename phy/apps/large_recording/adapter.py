"""Manifest-gated template controller for bounded recording evidence."""

from copy import deepcopy
from pathlib import Path

import numpy as np
from phylib.io.model import get_template_params
from phylib.utils import Bunch

from phy.cluster._propositions import MergePropositionController, decode_review_mapping

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
        controller = super()._load_merge_propositions()
        reviews = self.bounded_session.extra_documents.get('curation_review.json')
        if controller is not None and reviews is not None:
            source, mapping = decode_review_mapping(reviews)
            if source.get('sha256') != self._merge_proposition_sha256:
                raise ValueError('Committed reviews belong to different merge propositions.')
            controller = MergePropositionController(controller.catalog._replace(reviews=mapping))
        return controller

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

    def _evidence_amplitudes(self, cluster_ids, load_all=False):
        active = [c for c in cluster_ids if c is not None]
        eligible = [self._window_ids(c) for c in active]
        counts = _allocate_spike_counts(
            [len(ids) for ids in eligible],
            per_cluster=None if load_all else self.n_spikes_amplitudes,
            total=None if load_all else self.n_spikes_amplitudes_total,
        )
        values = self.bounded_session.attributes['amplitudes.npy'].reshape(-1)
        output = []
        if None in cluster_ids:
            ids = self._window_ids()
            ids = ids[~np.isin(self.supervisor.clustering.spike_clusters[ids], active)]
            ids = _sample_spikes_evenly(ids, self.n_spikes_amplitudes_background)
            ids = ids[np.isfinite(values[ids])]
            output.append(
                Bunch(amplitudes=values[ids], spike_ids=ids, spike_times=self._window_time(ids))
            )
        for ids, count in zip(eligible, counts):
            ids = _sample_spikes_evenly(ids, int(count))
            ids = ids[np.isfinite(values[ids])]
            output.append(
                Bunch(amplitudes=values[ids], spike_ids=ids, spike_times=self._window_time(ids))
            )
        return output

    def create_evidence_amplitude_view(self):
        array = self.bounded_session.attributes.get('amplitudes.npy')
        if array is None or (array.ndim != 1 and array.shape[1:] != (1,)):
            return
        start, end = self.current_evidence_window
        view = EvidenceAmplitudeView(
            amplitudes={'source': self._evidence_amplitudes},
            amplitudes_type='source',
            duration=(end - start) / self.model.sample_rate,
            split_is_eligible=lambda: False,
        )
        view.ex_status = self._window_status()
        view.set_window_context(
            self.evidence_window_index,
            start / self.model.sample_rate,
            end / self.model.sample_rate,
        )
        return view

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
        gui = getattr(self.supervisor, 'gui', None)
        if gui is not None:
            start, end = self.current_evidence_window
            for view in gui.list_views('EvidenceAmplitudeView', 'FeatureView'):
                view.ex_status = self._window_status()
                if isinstance(view, EvidenceAmplitudeView):
                    view.duration = (end - start) / self.model.sample_rate
                    view.set_window_context(
                        index, start / self.model.sample_rate, end / self.model.sample_rate
                    )
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
            split_action.setEnabled(False)
            split_action.setToolTip(
                'Selected-row splits cannot assign unseen source spikes; use a source-amplitude predicate.'
            )
        predicate_action = self.supervisor.actions.get('Split by source amplitude')
        if predicate_action is not None:
            predicate_action.setEnabled(
                'amplitudes.npy' in self.bounded_session.capabilities['predicate_split_attributes']
            )

        return gui
