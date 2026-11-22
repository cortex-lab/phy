"""Guarded native commands paired with semantic state, history and checkpoints."""

from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from types import MappingProxyType

import numpy as np
from phylib.utils import emit

from phy.cluster._history import History
from phy.cluster._utils import ClusterMeta
from phy.cluster.clustering import Clustering, _concatenate_spike_clusters
from phy.cluster.supervisor import Supervisor, _is_group_masked

from .curation import evaluate_predicate, interpret_events


def _table_values_equal(previous, current):
    """Compare display values without assuming a scalar metric result."""
    if isinstance(previous, np.ndarray) or isinstance(current, np.ndarray):
        for equal_nan in (True, False):
            try:
                return bool(np.array_equal(previous, current, equal_nan=equal_nan))
            except (TypeError, ValueError):
                # Object/string arrays do not necessarily support isnan. If a
                # nested object also has ambiguous equality, refresh its value.
                continue
        return False
    try:
        return bool(previous == current)
    except (TypeError, ValueError):
        return False


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
        result.flags.writeable = bool(getattr(self, '_internal_merge_write', False))
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

    def _do_merge(self, *args, **kwargs):
        # The native primitive writes through its public property. Restrict that
        # permission to the primitive itself, before any observer is called.
        self._internal_merge_write = True
        try:
            return super()._do_merge(*args, **kwargs)
        finally:
            self._internal_merge_write = False

    def merge(self, *args, **kwargs):
        self.policy.require()
        return super().merge(*args, **kwargs)

    def restore_partition(self, *args, **kwargs):
        self.policy.require()
        return super().restore_partition(*args, **kwargs)

    def assign(self, *args, **kwargs):
        self.policy.require()
        return super().assign(*args, **kwargs)

    def reset(self):
        self.policy.require()
        return super().reset()

    def undo(self):
        self.policy.require()
        transition = getattr(self, '_bounded_transition', lambda: None)()
        if transition is None:
            return super().undo()
        item = self._undo_stack.back()
        if item is None:
            return None
        _, _, undo_state = item
        pairs = [
            (rows, np.full(len(rows), unit, dtype=np.int64))
            for unit, rows in transition.old_groups.items()
            if len(rows)
        ]
        spike_ids, clusters = _concatenate_spike_clusters(*pairs)
        up = self._do_assign(spike_ids, clusters)
        up.history, up.undo_state = 'undo', undo_state
        emit('cluster', self, up)
        return up

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
        self._settle_selection()
        with self._transaction():
            return method(self, *args, **kwargs)

    return execute


class BoundedSupervisor(Supervisor):
    """Native UI/history operations paired with a validated semantic journal."""

    def __init__(self, *args, session, metadata_independent_metrics=None, **kwargs):
        self.session = session
        self._metadata_independent_metrics = dict(metadata_independent_metrics or {})
        self._bounded_table_revision = session.membership_revision
        self._bounded_table_metadata = session.metadata
        self._bounded_metadata_ids = set()
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
        self._metadata_independent_metrics['n_spikes'] = self.cluster_metrics['n_spikes']
        self.session_history = SessionHistory(session)
        self._saved_data_events = tuple(session.events)
        self.clustering._bounded_transition = self._native_undo_transition
        self.parent_origins.update(session.restored_parent_ids)
        self._proposition_projection_key = None
        self._validate_native(metadata=False, full=True)
        # Native construction already made the same immutable row index.
        self.session._assignment_groups = dict(self.clustering.spikes_per_cluster)

    def _native_undo_transition(self):
        events = self.session.events
        if not events:
            return None
        transition = self.session._transitions.get(id(events[-1]))
        if (
            transition is None
            or transition.after is not self.session._runtime
            or not transition.old_groups
        ):
            return None
        return transition

    def recovery_token(self):
        reviews = (
            tuple(self.merge_propositions.catalog.reviews.items())
            if self.merge_propositions is not None
            else self.session.extra_documents.get('curation_review.json')
        )
        return (
            tuple(self.session.events),
            max(self.session.next_cluster_id, self.clustering.new_cluster_id()),
            reviews,
        )

    @contextmanager
    def _transaction(self):
        self._recovery_depth = getattr(self, '_recovery_depth', 0) + 1
        try:
            with self._transaction_state():
                yield
        finally:
            self._recovery_depth -= 1
            recovery = getattr(self, 'recovery', None)
            if self._recovery_depth == 0 and recovery is not None:
                # Failed commands may consume allocator IDs, but their provisional
                # assignments/reviews have already been rolled back at this point.
                recovery.observe()

    @contextmanager
    def _transaction_state(self):
        """Roll native and semantic state back if an observer or command fails."""
        clustering = self.clustering
        arrays = (
            clustering._spike_clusters,
            dict(clustering._spikes_per_cluster),
            clustering._cluster_ids.copy(),
        )
        data, fields = deepcopy(self.cluster_meta._data), dict(self.cluster_meta._fields)
        parent_origins = dict(self.parent_origins)
        restored_partitions = dict(clustering._restored_partitions)
        lineage = self.parent_lineage
        lineage_state = (
            (
                dict(lineage.records),
                dict(lineage.color_records),
                dict(lineage.origins),
                dict(lineage.restorations),
            )
            if lineage is not None
            else None
        )
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
            # The old per-unit row arrays are immutable. Reconstruct on failure
            # instead of copying the complete assignment vector on every command.
            for cluster, rows in arrays[1].items():
                clustering._spike_clusters[rows] = cluster
            self.cluster_meta._data, self.cluster_meta._fields = data, fields
            self.parent_origins = parent_origins
            clustering._restored_partitions = restored_partitions
            if lineage_state is not None:
                (lineage.records, lineage.color_records, lineage.origins, lineage.restorations) = (
                    lineage_state
                )
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

    def _restore_history_context(self, selection, workflow_context, direction):
        if self.gui is not None:
            return super()._restore_history_context(selection, workflow_context, direction)
        self.selection.restore(selection)

    def _reset_cluster_view(self, *args, **kwargs):
        if self.gui is not None:
            return super()._reset_cluster_view(*args, **kwargs)

    def _workflow_context(self):
        return super()._workflow_context() if self.gui is not None else {}

    def _validate_native(self, metadata=True, full=False):
        # Local patches and the maintained nonempty catalog prove a normal
        # transition without scanning unrelated spike assignments. Save retains
        # the complete independent semantic projection and array comparison.
        if full or getattr(self, 'full_validation', False):
            valid = np.array_equal(self.clustering.spike_clusters, self.session.assignments)
        else:
            groups = self.session._assignment_groups
            native = self.clustering.spikes_per_cluster
            valid = set(native) == set(groups)
            if valid:
                for cluster, rows in self.session.last_changed_groups.items():
                    if not len(rows):
                        continue
                    if not np.array_equal(native.get(cluster), rows) or not np.all(
                        self.clustering.spike_clusters[rows] == cluster
                    ):
                        valid = False
                        break
        if not valid:
            raise ValueError('Native assignments disagree with the bounded semantic command.')
        if metadata:
            active, expected = self.session._runtime.active, self.session.metadata
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

    def _reassembled_results(self):
        """Proof of reassembly from immutable expression scopes, never spike scans."""
        from phy.cluster._propositions import ReviewDecision

        from .live_summaries import _membership_contains, membership_expressions
        from .source import hash_json

        if self.merge_propositions is None:
            return {}
        history = {}
        active = membership_expressions(
            self.session.unit_ids, self.session.events, history=history
        )
        historical_by_unit = {unit: expression for (_, unit), expression in history.items()}

        live = set(self.clustering.cluster_ids)
        current = {unit: expression for unit, expression in active.items() if unit in live}
        atom_cache, proof_cache = {}, {}

        def atoms(expression):
            key = id(expression)
            if key not in atom_cache:
                parts = expression['parts'] if expression['kind'] == 'union' else [expression]
                atom_cache[key] = frozenset(hash_json(part) for part in parts)
            return atom_cache[key]

        result = {}
        catalog = self.merge_propositions.catalog
        for key, review in catalog.reviews.items():
            if review.decision is not ReviewDecision.ACCEPTED:
                continue
            expression = historical_by_unit.get(review.result_unit_id)
            if expression is None or not catalog._result_was_restored(review.result_unit_id):
                continue
            expected_atoms = atoms(expression)
            matches = [
                (unit, expression != scope)
                for unit, scope in current.items()
                if expected_atoms <= atoms(scope)
                or _membership_contains(scope, expression, proof_cache)
            ]
            # Distinct live memberships must never both prove the same full scope.
            if len(matches) == 1:
                unit, modified = matches[0]
                result[key] = {'result_unit_id': unit, 'modified': modified}
        return result

    def _project_proposition_state(self):
        if self.merge_propositions is not None:
            key = (
                self.merge_propositions,
                self.session.membership_revision,
                tuple(map(int, self.clustering.cluster_ids)),
                tuple(self.merge_propositions.catalog.reviews.items()),
            )
            if key == getattr(self, '_proposition_projection_key', None):
                return
            self.merge_propositions.project_live_ids(
                self.clustering.cluster_ids,
                retired_units=self.session.retired_units,
                reassembled_results=self._reassembled_results(),
            )
            self._proposition_projection_key = key

    def _cluster_metadata_changed(self, field, cluster_ids):
        # Native metadata notifications still update filters, sorting, masking,
        # selection, and plugin listeners synchronously. Retain their row scope
        # for the final semantic refresh, including label undo/redo.
        cluster_ids = list(cluster_ids)
        self._bounded_metadata_ids.update(cluster_ids)
        if self.gui is not None:
            return super()._cluster_metadata_changed(field, cluster_ids)

    def _metadata_refresh_ids(self):
        changed = set(self._bounded_metadata_ids)
        previous, current = self._bounded_table_metadata, self.session.metadata
        for field in set(previous) | set(current):
            before, after = previous.get(field, {}), current.get(field, {})
            changed.update(
                cluster
                for cluster in set(before) | set(after)
                if before.get(cluster) != after.get(cluster)
            )
        return sorted(changed & set(self.clustering.cluster_ids))

    def _metadata_refresh_rows(self):
        changed = set(self._metadata_refresh_ids())
        custom = {
            name: provider
            for name, provider in self.cluster_metrics.items()
            if self._metadata_independent_metrics.get(name) is not provider
        }
        authoritative = {
            'n_spikes',
            'fr',
            'full_count',
            'evidence_count',
            'waveform_count',
            'evidence',
        }
        rows = []
        for cluster in self.clustering.cluster_ids:
            if cluster in changed:
                rows.append(self.get_cluster_info(cluster))
            elif custom:
                row = {'id': cluster}
                for name, provider in custom.items():
                    value = provider(cluster)
                    # Match get_cluster_info's producer-metadata precedence and
                    # authoritative bounded count/rate/masking fields exactly.
                    if name in self.cluster_meta.fields and name not in authoritative:
                        value = self.cluster_meta.get(name, cluster)
                    if name == 'is_masked':
                        value = _is_group_masked(self.cluster_meta.get('group', cluster))
                    row[name] = value
                rows.append(row)
        return rows

    def _refresh_bounded(self):
        self.parent_origins = self.session.restored_parent_ids
        if self.merge_propositions is not None:
            self._project_proposition_state()
            self._refresh_propositions()
        if self.gui is not None:
            metadata_only = self._bounded_table_revision is self.session.membership_revision
            data = self._metadata_refresh_rows() if metadata_only else self.cluster_info
            for view in (self.cluster_view, self.similarity_view, self.merge_view):
                if view is None:
                    continue
                if metadata_only:
                    patches = []
                    for row in data:
                        previous = view.get(row['id'])
                        if previous is None:
                            continue
                        patch = {
                            key: value
                            for key, value in row.items()
                            if key not in previous or not _table_values_equal(previous[key], value)
                        }
                        if patch:
                            patches.append({'id': row['id'], **patch})
                    view.change(patches)
                else:
                    view.change(data)
        self._bounded_table_revision = self.session.membership_revision
        self._bounded_table_metadata = self.session.metadata
        self._bounded_metadata_ids.clear()
        emit('bounded_state_changed', self)

    def _capture(self):
        if self.session.metadata_policy:
            self._synchronize_metadata()
        self._validate_native()
        self.session_history.capture()
        self._global_history.add_to_current_action(self.session_history)
        self._is_dirty = tuple(self.session.events) != self._saved_data_events
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
    def restore_parents(self, target=None, *, groups=None, colors=None, new_cluster_ids=None):
        """Restore complete semantic parents, then edit their fresh live IDs."""
        self._require_capability('merge')
        if target is None:
            selected = list(self.selected)
            if len(selected) != 1:
                raise ValueError('Select one current merged unit to restore its parents.')
            target = selected[0]
        target = int(target)
        _, expected = self.session.restore_parent_groups(target, recursive=True)
        if groups is not None and (
            list(groups) != list(expected)
            or any(not np.array_equal(groups[parent], rows) for parent, rows in expected.items())
        ):
            raise ValueError('Restored groups must match the complete semantic parent partition.')
        if any(not len(rows) for rows in expected.values()):
            raise ValueError(
                'A complete source parent has no retained evidence; restoration requires '
                'evidence for every parent and cannot omit scientific units.'
            )
        first = max(self.clustering.new_cluster_id(), self.session.next_cluster_id)
        children = (
            list(range(first, first + len(expected)))
            if new_cluster_ids is None
            else list(new_cluster_ids)
        )
        event = self.session.preflight_restore_parents(target, children, recursive=True)
        with self.policy.command():
            out = super().restore_parents(
                target,
                groups=expected,
                colors=colors,
                new_cluster_ids=children,
            )
        if out is None:
            return None
        self.session._append(event)
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
        # Validate intent without mutating either semantic or native state.
        event = self.session.preflight_label(name, ids, value)
        with self.policy.command():
            super().label(name, value, cluster_ids=ids)
        self.session._append(event)
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
        self._settle_selection()
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
        event = self.session.preflight_split(
            int(cluster_id), selected_id, remainder_id, predicate, metadata=metadata
        )
        with self.policy.command():
            out = Supervisor.split(self, selected)
        self.session._append(event)
        self._capture()
        return out

    @semantic_command
    def _reject_merge_proposition(self, sender, key):
        return super()._reject_merge_proposition(sender, key)

    @semantic_command
    def _reset_merge_proposition(self, sender, key):
        return super()._reset_merge_proposition(sender, key)

    def adopt_recovery(self, recovered):
        """Adopt a fully prepared draft before attaching GUI views; new undo baseline."""
        from phy.cluster._history import CurationHistoryEntry
        from phy.cluster._propositions import decode_review_mapping
        from phy.cluster._selection import CurationSelectionController

        if self.gui is not None and not self.gui._closed:
            raise RuntimeError('Restore a recovery draft before creating curation views.')
        if (
            recovered.assignments is not recovered._certified_assignments
            or recovered._recovery_clustering._spike_clusters is not recovered._certified_native
        ):
            raise ValueError('Recovered assignment certification changed before adoption.')
        session = self.session
        for name in (
            'assignments',
            'metadata',
            '_events',
            'next_cluster_id',
            'extra_documents',
            '_runtime',
            '_assignment_groups',
            '_transitions',
            '_transition_bytes',
            '_prepared',
            '_catalog_cache',
            'last_changed_groups',
        ):
            setattr(session, name, getattr(recovered, name))
        native = recovered._recovery_clustering
        for name, value in vars(native).items():
            if name != 'policy':
                setattr(self.clustering, name, value)
        self.clustering._bounded_transition = self._native_undo_transition
        self.cluster_meta._undo_stack.clear()
        self._synchronize_metadata()
        self._global_history.clear(CurationHistoryEntry())
        self.selection = CurationSelectionController()
        self.session_history = SessionHistory(session)
        if self.merge_propositions is not None:
            _, reviews = decode_review_mapping(session.extra_documents['curation_review.json'])
            catalog = self.merge_propositions.catalog._replace(reviews=reviews)
            self.merge_propositions.restore(catalog)
            self.merge_propositions._history.clear(catalog)
        self.parent_origins = session.restored_parent_ids
        self._proposition_projection_key = None
        self._project_proposition_state()
        self._bounded_table_revision = session.membership_revision
        self._bounded_table_metadata = session.metadata
        self._bounded_metadata_ids.clear()
        self._is_dirty = True
        self._validate_native()

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
        self._is_dirty = tuple(self.session.events) != self._saved_data_events
        self._update_save_feedback()
        self._synchronize_metadata()
        self._refresh_bounded()

    def save(self):
        recovery = getattr(self, 'recovery', None)
        token = recovery.checkpoint_token() if recovery is not None else None
        try:
            self._save_checkpoint()
        except Exception as error:
            if recovery is not None:
                recovery.checkpoint_failed(error)
            raise
        if recovery is not None:
            recovery.checkpoint_saved(token)

    def _save_checkpoint(self):
        self._settle_selection()
        saved_state = self.recovery_token()
        saved_catalog = (
            self.merge_propositions.catalog if self.merge_propositions is not None else None
        )
        self._validate_native(full=True)
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
        if saved_catalog is not None:
            self.merge_propositions._saved_reviews = self.merge_propositions._review_fingerprint(
                saved_catalog
            )
        self._saved_data_events = saved_state[0]
        self._is_dirty = tuple(self.session.events) != self._saved_data_events
        self._update_save_feedback(saved=not self.is_dirty())
        emit('curation_saved', self)
