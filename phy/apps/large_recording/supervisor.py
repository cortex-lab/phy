"""Guarded native commands paired with semantic state, history and checkpoints."""

from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from types import MappingProxyType

import numpy as np
from phylib.utils import emit

from phy.cluster._history import History
from phy.cluster._utils import ClusterMeta
from phy.cluster.clustering import Clustering
from phy.cluster.supervisor import Supervisor

from .curation import evaluate_predicate, interpret_events


class MutationPolicy:
    """Only semantic commands may mutate native bounded state."""

    def __init__(self):
        self.depth = 0

    def require(self):
        if not self.depth:
            raise ValueError(
                'Bounded curation requires a semantic command. Use supervisor.merge(), '
                'supervisor.label(), or split_source_amplitude(); arbitrary splits are unavailable.'
            )

    @contextmanager
    def command(self):
        self.depth += 1
        try:
            yield
        finally:
            self.depth -= 1


class GuardedClustering(Clustering):
    def __init__(self, *args, policy, **kwargs):
        self.policy = policy
        super().__init__(*args, **kwargs)

    @property
    def spike_clusters(self):
        result = self._spike_clusters.view()
        result.flags.writeable = bool(self.policy.depth)
        return result

    @property
    def cluster_ids(self):
        result = self._cluster_ids.view()
        result.flags.writeable = False
        return result

    @property
    def spikes_per_cluster(self):
        return MappingProxyType(self._spikes_per_cluster)

    def _update_cluster_ids(self, *args, **kwargs):
        super()._update_cluster_ids(*args, **kwargs)
        for ids in self._spikes_per_cluster.values():
            ids.flags.writeable = False

    def merge(self, *args, **kwargs):
        self.policy.require()
        return super().merge(*args, **kwargs)

    def assign(self, *args, **kwargs):
        self.policy.require()
        return super().assign(*args, **kwargs)

    def reset(self):
        self.policy.require()
        return super().reset()

    def undo(self):
        self.policy.require()
        return super().undo()

    def redo(self):
        self.policy.require()
        return super().redo()


class GuardedMeta(ClusterMeta):
    def __init__(self, policy):
        self.policy = policy
        super().__init__()

    def set(self, *args, **kwargs):
        self.policy.require()
        return super().set(*args, **kwargs)

    def add_field(self, *args, **kwargs):
        self.policy.require()
        return super().add_field(*args, **kwargs)

    def from_dict(self, *args, **kwargs):
        self.policy.require()
        return super().from_dict(*args, **kwargs)

    def undo(self):
        self.policy.require()
        return super().undo()

    def redo(self):
        self.policy.require()
        return super().redo()


class SessionHistory(History):
    def __init__(self, session):
        self.session = session
        super().__init__(session.journal_state())

    def capture(self):
        self.add(self.session.journal_state())

    def undo(self):
        self.back()
        self.session.restore_journal(self.current_item)

    def redo(self):
        self.forward()
        self.session.restore_journal(self.current_item)


def semantic_command(method):
    @wraps(method)
    def execute(self, *args, **kwargs):
        with self._transaction():
            return method(self, *args, **kwargs)

    return execute


class BoundedSupervisor(Supervisor):
    """Native UI/history operations paired with a validated semantic journal."""

    def __init__(self, *args, session, **kwargs):
        self.session = session
        self.policy = MutationPolicy()
        self.select_actions = None
        self.cluster_view = None
        self.similarity_view = None

        def clustering_factory(*args, **kwargs):
            return GuardedClustering(*args, policy=self.policy, **kwargs)

        def meta_factory(groups, labels):
            meta = GuardedMeta(self.policy)
            data = {}
            for field, values in labels.items():
                for cluster, value in values.items():
                    data.setdefault(cluster, {})[field] = value
            with self.policy.command():
                meta.add_field('group')
                meta.from_dict(data)
            return meta

        # Generation state supersedes disposable clustering caches.
        kwargs['context'] = None
        kwargs['new_cluster_id'] = session.next_cluster_id
        kwargs['clustering_factory'] = clustering_factory
        kwargs['cluster_meta_factory'] = meta_factory
        super().__init__(*args, **kwargs)
        self.session_history = SessionHistory(session)

    @contextmanager
    def _transaction(self):
        """Roll native and semantic state back if an observer or command fails."""
        clustering = self.clustering
        arrays = (
            clustering._spike_clusters.copy(),
            dict(clustering._spikes_per_cluster),
            clustering._cluster_ids.copy(),
        )
        data, fields = deepcopy(self.cluster_meta._data), dict(self.cluster_meta._fields)
        histories = [
            clustering._undo_stack,
            self.cluster_meta._undo_stack,
            self._global_history,
            self.session_history,
        ]
        if self.merge_propositions is not None:
            histories.append(self.merge_propositions._history)
        history_states = [(list(h._history), h._index) for h in histories]
        journal, selection, dirty = (
            self.session.journal_state(),
            self.selection.snapshot(),
            self._is_dirty,
        )
        propositions = (
            self.merge_propositions.snapshot() if self.merge_propositions is not None else None
        )
        workflow, columns = self._workflow_context(), list(self.columns)
        try:
            yield
        except Exception:
            clustering._spike_clusters, clustering._spikes_per_cluster, clustering._cluster_ids = (
                arrays
            )
            self.cluster_meta._data, self.cluster_meta._fields = data, fields
            for history, (items, index) in zip(histories, history_states):
                history._history, history._index = items, index
            self.session.restore_journal(journal)
            self.session.next_cluster_id = max(
                self.session.next_cluster_id, clustering.new_cluster_id()
            )
            self.selection.restore(selection)
            if propositions is not None:
                self.merge_propositions.restore(propositions)
            self._is_dirty = dirty
            self.columns = columns
            self._lineage_cache = None
            if self.gui is not None:
                self._reset_cluster_view()
                self._restore_history_context(
                    selection,
                    {
                        'mode': 'merge' if selection.is_merge_mode else 'post_merge',
                        'tables': workflow,
                    },
                    'undo',
                )
                self._refresh_bounded()
                self._update_save_feedback()
            raise

    def _require_capability(self, name):
        if not self.session.manifest.get('capabilities', {}).get(name, False):
            raise ValueError(f'This bounded package does not allow {name} operations.')

    def get_cluster_info(self, cluster_id, exclude=()):
        info = super().get_cluster_info(cluster_id, exclude=exclude)
        # Producer TSV labels may use metric names. They must not overwrite
        # authoritative complete-sorting counts/rates with inherited values.
        for name in (
            'n_spikes',
            'fr',
            'full_count',
            'evidence_count',
            'waveform_count',
            'evidence',
        ):
            if name not in exclude and name in self.cluster_metrics:
                info[name] = self.cluster_metrics[name](cluster_id)
        return info

    def _select_after_merge(self, *args, **kwargs):
        if self.gui is not None:
            return super()._select_after_merge(*args, **kwargs)

    def _select_after_split(self, *args, **kwargs):
        if self.gui is not None:
            return super()._select_after_split(*args, **kwargs)

    def _restore_history_context(self, *args, **kwargs):
        if self.gui is not None:
            return super()._restore_history_context(*args, **kwargs)

    def _reset_cluster_view(self, *args, **kwargs):
        if self.gui is not None:
            return super()._reset_cluster_view(*args, **kwargs)

    def _workflow_context(self):
        return super()._workflow_context() if self.gui is not None else {}

    def _validate_native(self, metadata=True):
        if not np.array_equal(self.clustering.spike_clusters, self.session.assignments):
            raise ValueError('Native assignments disagree with the bounded semantic command.')
        if metadata:
            active, expected, _ = interpret_events(
                self.session.unit_ids, self.session.base_metadata, self.session.events
            )
            for field in set(self.cluster_meta.fields) | set(expected):
                for cluster in active:
                    actual = (
                        self.cluster_meta.get(field, cluster)
                        if field in self.cluster_meta.fields
                        else None
                    )
                    if actual != expected.get(field, {}).get(cluster):
                        raise ValueError(
                            'Native metadata disagrees with the bounded semantic command.'
                        )

    def _synchronize_metadata(self):
        # Native ClusterMeta.undo does not retain nonstacked inherited merge
        # labels. The semantic projection is authoritative in bounded mode.
        data = {}
        for field, values in self.session.metadata.items():
            if field not in self.cluster_meta.fields:
                with self.policy.command():
                    self.cluster_meta.add_field(field)
            for cluster, value in values.items():
                data.setdefault(cluster, {})[field] = value
        self.cluster_meta._data = data

    def _refresh_bounded(self):
        if self.merge_propositions is not None:
            self.merge_propositions.project_live_ids(
                self.clustering.cluster_ids, retired_units=self.session.retired_units
            )
        if self.gui is not None:
            data = self.cluster_info
            self.cluster_view.change(data)
            self.similarity_view.change(data)
            if self.merge_view is not None:
                self.merge_view.change(data)
        emit('bounded_state_changed', self)

    def _capture(self):
        if self.session.metadata_policy:
            self._synchronize_metadata()
        self._validate_native()
        self.session_history.capture()
        self._global_history.add_to_current_action(self.session_history)
        self._is_dirty = True
        self._update_save_feedback()
        self._synchronize_metadata()
        self._refresh_bounded()

    @semantic_command
    def merge(self, cluster_ids=None, to=None):
        self._require_capability('merge')
        parents = list(self.selected if cluster_ids is None else cluster_ids)
        if len(parents) < 2:
            return
        target = max(self.clustering.new_cluster_id(), self.session.next_cluster_id)
        target = target if to is None else int(to)
        self.session.preflight_merge(parents, target)
        with self.policy.command():
            out = super().merge(cluster_ids=cluster_ids, to=target)
        if out is None:
            return
        resolved = {
            field: self.cluster_meta.get(field, target) for field in self.cluster_meta.fields
        }
        self.session.record_merge(parents, target, metadata=resolved)
        self._capture()
        return out

    @semantic_command
    def label(self, name, value, cluster_ids=None):
        self._require_capability('label')
        if name in (
            'n_spikes',
            'fr',
            'full_count',
            'evidence_count',
            'waveform_count',
            'evidence',
        ):
            raise ValueError('Bounded count/rate/evidence columns are read-only metrics.')
        ids = list(self.selected if cluster_ids is None else np.atleast_1d(cluster_ids))
        if not ids or self.selection.state.is_merge_mode:
            return
        # Preflight against a copy before touching native metadata.
        before = self.session.snapshot()
        self.session.record_label(name, ids, value)
        self.session.restore(before)
        with self.policy.command():
            super().label(name, value, cluster_ids=ids)
        self.session.record_label(name, ids, value)
        self._capture()

    def _graphical_split_views(self):
        if self.gui is None:
            return []
        return [
            view
            for view in self.gui.list_views('EvidenceAmplitudeView')
            if not view._closed
            and view.gui is self.gui
            and view._dock_visible
            and view.split_threshold is not None
            and np.isfinite(view.split_threshold)
            and list(view.cluster_ids) == list(self.selected)
            and len(self.selected) == 1
            and not self.selection.state.is_merge_mode
            and view.amplitudes_type == 'source'
        ]

    def _update_graphical_split_action(self):
        actions = getattr(self, 'actions', None)
        action = actions.get('split') if actions is not None else None
        if action is not None:
            capable = 'amplitudes.npy' in self.session.capabilities['predicate_split_attributes']
            action.setEnabled(capable and len(self._graphical_split_views()) == 1)

    def _set_merge_mode_ui(self, active):
        if self.gui is None:
            return
        super()._set_merge_mode_ui(active)
        if active:
            for view in self.gui.list_views('EvidenceAmplitudeView'):
                view.clear_amplitude_split_threshold()
        self._update_graphical_split_action()

    def split(self, spike_ids=None, spike_clusters_rel=0):
        if spike_ids is not None or spike_clusters_rel != 0:
            raise ValueError(
                'Lasso, feature and waveform splits cannot be replayed on unseen spikes. '
                'Use Split by source amplitude for a full-source scalar predicate.'
            )
        views = self._graphical_split_views()
        if len(views) != 1:
            raise ValueError('Exactly one visible source-amplitude threshold preview is required.')
        view = views[0]
        threshold = view.split_threshold
        cluster_id = int(self.selected[0])
        try:
            out = self.split_source_amplitude(cluster_id, threshold)
        except Exception as error:
            # A transaction may have published transient native selection updates
            # before rollback. Keep the user's predicate available for correction.
            view.split_threshold = threshold
            view._replot_displayed_amplitudes()
            emit(
                'amplitude_split_preview_changed',
                view,
                cluster_id=cluster_id,
                amplitudes_type='source',
                threshold=threshold,
            )
            view._show_split_status(str(error))
            raise
        view.clear_split_selection()
        self._update_graphical_split_action()
        return out

    @semantic_command
    def split_source_amplitude(self, cluster_id, threshold):
        if 'amplitudes.npy' not in self.session.manifest.get('capabilities', {}).get(
            'predicate_split_attributes', []
        ):
            raise ValueError('This package does not support source-amplitude predicate splits.')
        if self.selection.state.is_merge_mode:
            raise ValueError('Leave Merge mode before splitting by source amplitude.')
        if 'amplitudes.npy' not in self.session.attributes:
            raise ValueError('The source has no scalar per-spike amplitudes.')
        predicate = {
            'attribute': 'amplitudes.npy',
            'operator': '<',
            'threshold': float(threshold),
            'nonfinite': 'keep',
        }
        ids = self.clustering.spikes_per_cluster.get(int(cluster_id), np.array([], dtype=np.int64))
        selected = ids[
            evaluate_predicate(self.session.attributes['amplitudes.npy'][ids], predicate)
        ]
        if not len(selected) or len(selected) == len(ids):
            raise ValueError('Both sides of the threshold must have evidence spikes.')
        selected_id = max(self.clustering.new_cluster_id(), self.session.next_cluster_id)
        remainder_id = selected_id + 1
        self.clustering._new_cluster_id = selected_id
        metadata = {
            field: {
                child: self.cluster_meta.get(field, cluster_id)
                for child in (selected_id, remainder_id)
            }
            for field in self.cluster_meta.fields
        }
        before = self.session.snapshot()
        self.session.record_threshold_split(
            int(cluster_id), [remainder_id, selected_id], predicate, metadata=metadata
        )
        self.session.restore(before)
        with self.policy.command():
            out = Supervisor.split(self, selected)
        self.session.record_threshold_split(
            int(cluster_id), [remainder_id, selected_id], predicate, metadata=metadata
        )
        self._capture()
        return out

    @semantic_command
    def undo(self):
        with self.policy.command():
            super().undo()
        self._after_history()

    @semantic_command
    def redo(self):
        with self.policy.command():
            super().redo()
        self._after_history()

    def _after_history(self):
        self._validate_native(metadata=False)
        # Never reuse allocated IDs after branching or restoring a snapshot.
        self.session.next_cluster_id = max(
            self.session.next_cluster_id, self.clustering.new_cluster_id()
        )
        self._is_dirty = True
        self._update_save_feedback()
        self._synchronize_metadata()
        self._refresh_bounded()

    def save(self):
        self._validate_native()
        self.session.validate_proposition_documents()
        documents = None
        if self.merge_propositions is not None:
            documents = dict(self.session.extra_documents)
            documents.update(
                {
                    'curation_review.json': self.merge_propositions.catalog.review_mapping(
                        source_sha256=self.session.proposition_documents['curation.json']['sha256']
                    )
                }
            )
        self.session.save(
            assignments=self.clustering.spike_clusters,
            metadata=self.session.metadata,
            next_cluster_id=max(self.session.next_cluster_id, self.clustering.new_cluster_id()),
            extra_documents=documents,
        )
        if self.merge_propositions is not None:
            self.merge_propositions.mark_saved()
        self._is_dirty = False
        self._update_save_feedback(saved=True)
