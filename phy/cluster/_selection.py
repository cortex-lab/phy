"""Immutable selection state used by curation workflows.

This module intentionally has no Qt or Supervisor dependencies.  Views can use
the controller as a synchronous source of selection state, while deciding
separately how and when to render a :class:`SelectionChange`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class WorkflowMode(Enum):
    """The active curation workflow."""

    NORMAL = 'normal'
    MERGE = 'merge'


def _as_unique_ids(cluster_ids) -> tuple[int, ...]:
    """Return *cluster_ids* as a tuple, rejecting duplicates."""
    cluster_ids = tuple(cluster_ids)
    if len(cluster_ids) != len(set(cluster_ids)):
        raise ValueError('Cluster IDs must be unique.')
    return cluster_ids


def _ordered_union(*cluster_id_lists) -> tuple[int, ...]:
    """Return the ordered union of the supplied cluster-ID sequences."""
    return tuple(dict.fromkeys(cluster_id for ids in cluster_id_lists for cluster_id in ids))


@dataclass(frozen=True)
class NormalWorkflowSnapshot:
    """Normal-mode selection plus opaque view state needed for cancellation."""

    cluster_ids: tuple[int, ...]
    similar_ids: tuple[int, ...]
    reference_id: int | None
    presentation_order: tuple[int, ...]
    workflow_context: object = None

    def __post_init__(self):
        state = CurationSelectionState(
            cluster_ids=self.cluster_ids,
            similar_ids=self.similar_ids,
            reference_id=self.reference_id,
            presentation_order=self.presentation_order,
        )
        object.__setattr__(self, 'cluster_ids', state.cluster_ids)
        object.__setattr__(self, 'similar_ids', state.similar_ids)
        object.__setattr__(self, 'reference_id', state.reference_id)
        object.__setattr__(self, 'presentation_order', state.presentation_order)

    @property
    def selection(self):
        """Return the Normal-mode selection represented by this snapshot."""
        return CurationSelectionState(
            cluster_ids=self.cluster_ids,
            similar_ids=self.similar_ids,
            reference_id=self.reference_id,
            presentation_order=self.presentation_order,
        )


@dataclass(frozen=True)
class MergeSession:
    """Temporary ordered merge workspace tied to one fixed reference cluster."""

    reference_id: int
    ordered_ids: tuple[int, ...]
    entry_snapshot: NormalWorkflowSnapshot

    def __post_init__(self):
        ordered_ids = _as_unique_ids(self.ordered_ids)
        if not ordered_ids or ordered_ids[0] != self.reference_id:
            raise ValueError('The merge reference must be the first staged cluster.')
        object.__setattr__(self, 'ordered_ids', ordered_ids)


@dataclass(frozen=True)
class CurationSelectionState:
    """The authoritative, immutable curation selection.

    ``presentation_order`` is the effective selection in the order delivered
    to scientific views. The Supervisor derives Normal-mode order from the
    visible role tables. In Merge mode it is derived from the visible roles:
    Merge View order first, followed by Similarity View selection order.
    """

    mode: WorkflowMode = WorkflowMode.NORMAL
    cluster_ids: tuple[int, ...] = ()
    similar_ids: tuple[int, ...] = ()
    reference_id: int | None = None
    presentation_order: tuple[int, ...] | None = None
    merge: MergeSession | None = None

    def __post_init__(self):
        cluster_ids = _as_unique_ids(self.cluster_ids)
        similar_ids = _as_unique_ids(self.similar_ids)
        reference_id = self.reference_id
        merge = self.merge
        if self.mode is WorkflowMode.NORMAL:
            if merge is not None:
                raise ValueError('Normal mode cannot contain a merge session.')
            if reference_id is None and cluster_ids:
                reference_id = cluster_ids[0]
            if reference_id is not None and reference_id not in cluster_ids:
                raise ValueError('The reference ID must belong to the cluster selection.')
            effective_ids = _ordered_union(cluster_ids, similar_ids)
        else:
            if merge is None:
                raise ValueError('Merge mode requires a merge session.')
            if cluster_ids:
                raise ValueError('Cluster selection must be empty in Merge mode.')
            if reference_id is None:
                reference_id = merge.reference_id
            if reference_id != merge.reference_id:
                raise ValueError('The selection and merge references must agree.')
            if set(similar_ids).intersection(merge.ordered_ids):
                raise ValueError('A cluster cannot be both staged and selected as similar.')
            effective_ids = _ordered_union(merge.ordered_ids, similar_ids)
        default_presentation = _ordered_union(
            (reference_id,) if reference_id is not None else (),
            merge.ordered_ids if merge is not None else cluster_ids,
            similar_ids,
        )
        presentation_order = (
            default_presentation
            if self.presentation_order is None or self.mode is WorkflowMode.MERGE
            else _as_unique_ids(self.presentation_order)
        )

        if set(presentation_order) != set(effective_ids):
            raise ValueError('Presentation order must contain exactly the effective IDs.')
        if (
            presentation_order
            and reference_id is not None
            and presentation_order[0] != reference_id
        ):
            raise ValueError('The reference ID must occupy the first presentation slot.')

        object.__setattr__(self, 'cluster_ids', cluster_ids)
        object.__setattr__(self, 'similar_ids', similar_ids)
        object.__setattr__(self, 'reference_id', reference_id)
        object.__setattr__(self, 'presentation_order', presentation_order)

    @property
    def effective_ids(self):
        """Return the effective selection for the active workflow mode."""
        return _ordered_union(self.merge_ids, self.similar_ids)

    @property
    def merge_ids(self):
        """Return staged IDs in Merge mode, otherwise the Cluster selection."""
        return self.merge.ordered_ids if self.merge is not None else self.cluster_ids

    @property
    def is_merge_mode(self):
        return self.mode is WorkflowMode.MERGE


# A state is itself an immutable and complete snapshot for Normal mode.  The
# alias makes the snapshot boundary explicit at controller call sites.
CurationSelectionSnapshot = CurationSelectionState


@dataclass(frozen=True)
class SelectionChange:
    """The complete before/after diff for one selection transition."""

    before: CurationSelectionState
    after: CurationSelectionState
    roles_changed: bool
    presentation_changed: bool
    reference_changed: bool
    mode_changed: bool

    @classmethod
    def create(cls, before, after):
        """Classify the transition from *before* to *after*."""
        return cls(
            before=before,
            after=after,
            roles_changed=(
                before.cluster_ids != after.cluster_ids
                or before.similar_ids != after.similar_ids
                or before.merge_ids != after.merge_ids
            ),
            presentation_changed=before.presentation_order != after.presentation_order,
            reference_changed=before.reference_id != after.reference_id,
            mode_changed=before.mode is not after.mode,
        )

    @property
    def changed(self):
        """Whether this transition changes any modeled state."""
        return self.before != self.after


class CurationSelectionController:
    """Apply validated, atomic curation selection transitions."""

    def __init__(self, state=None):
        self._state = state or CurationSelectionState()

    @property
    def state(self):
        """Return the current immutable selection state."""
        return self._state

    def snapshot(self):
        """Return the current immutable selection state."""
        return self._state

    def restore(self, snapshot):
        """Restore a previously captured Normal-mode *snapshot*."""
        if not isinstance(snapshot, CurationSelectionState):
            raise TypeError('Expected a CurationSelectionState snapshot.')
        return self._apply(snapshot)

    def set_cluster_selection(self, cluster_ids, reference_id=None):
        """Set Cluster View IDs, using the first (blue) ID as the default reference."""
        self._require_normal_mode()
        cluster_ids = _as_unique_ids(cluster_ids)
        if reference_id is None:
            reference_id = cluster_ids[0] if cluster_ids else None
        after = CurationSelectionState(
            cluster_ids=cluster_ids,
            similar_ids=self._state.similar_ids,
            reference_id=reference_id,
        )
        return self._apply(after)

    def set_normal_selection(
        self,
        cluster_ids,
        similar_ids=(),
        reference_id=None,
        presentation_order=None,
    ):
        """Atomically replace all Normal-mode selection roles and presentation state."""
        after = CurationSelectionState(
            cluster_ids=_as_unique_ids(cluster_ids),
            similar_ids=_as_unique_ids(similar_ids),
            reference_id=reference_id,
            presentation_order=presentation_order,
        )
        return self._apply(after)

    def set_similarity_selection(self, similar_ids):
        """Set Similarity View IDs without changing the current reference."""
        current = self._state
        similar_ids = _as_unique_ids(similar_ids)
        effective_ids = _ordered_union(current.merge_ids, similar_ids)
        presentation_order = _ordered_union(
            tuple(
                cluster_id
                for cluster_id in current.presentation_order
                if cluster_id in effective_ids
            ),
            effective_ids,
        )
        after = CurationSelectionState(
            mode=current.mode,
            cluster_ids=current.cluster_ids,
            similar_ids=similar_ids,
            reference_id=current.reference_id,
            presentation_order=presentation_order,
            merge=current.merge,
        )
        return self._apply(after)

    def clear_similarity_selection(self):
        """Clear only the Similarity View selection."""
        return self.set_similarity_selection(())

    def enter_merge_mode(self, workflow_context=None):
        """Stage the complete Normal-mode selection and enter Merge mode."""
        self._require_normal_mode()
        current = self._state
        if not current.cluster_ids:
            raise ValueError('Merge mode requires a Cluster View selection.')
        snapshot = NormalWorkflowSnapshot(
            cluster_ids=current.cluster_ids,
            similar_ids=current.similar_ids,
            reference_id=current.reference_id,
            presentation_order=current.presentation_order,
            workflow_context=workflow_context,
        )
        ordered_ids = current.presentation_order
        merge = MergeSession(current.reference_id, ordered_ids, snapshot)
        after = CurationSelectionState(
            mode=WorkflowMode.MERGE,
            reference_id=current.reference_id,
            presentation_order=current.presentation_order,
            merge=merge,
        )
        return self._apply(after)

    def cancel_merge_mode(self):
        """Leave Merge mode and restore the exact entry selection."""
        self._require_merge_mode()
        return self._apply(self._state.merge.entry_snapshot.selection)

    def add_to_merge(self, cluster_ids, insertion=None):
        """Stage candidates, removing them from Similarity selection if necessary."""
        self._require_merge_mode()
        cluster_ids = _as_unique_ids(cluster_ids)
        current = self._state
        new_ids = tuple(
            cluster_id for cluster_id in cluster_ids if cluster_id not in current.merge_ids
        )
        if not new_ids:
            return self._apply(current)
        ordered_ids = list(current.merge_ids)
        if insertion is None:
            insertion = len(ordered_ids)
        if not 1 <= insertion <= len(ordered_ids):
            raise ValueError('Merge insertion must follow the fixed reference.')
        ordered_ids[insertion:insertion] = new_ids
        merge = MergeSession(
            current.reference_id,
            tuple(ordered_ids),
            current.merge.entry_snapshot,
        )
        similar_ids = tuple(
            cluster_id for cluster_id in current.similar_ids if cluster_id not in new_ids
        )
        after = CurationSelectionState(
            mode=WorkflowMode.MERGE,
            similar_ids=similar_ids,
            reference_id=current.reference_id,
            merge=merge,
        )
        return self._apply(after)

    def remove_from_merge(self, cluster_ids):
        """Return staged non-reference candidates to the Similarity selection."""
        self._require_merge_mode()
        cluster_ids = _as_unique_ids(cluster_ids)
        current = self._state
        if current.reference_id in cluster_ids:
            raise ValueError('The merge reference cannot be removed.')
        if not set(cluster_ids) <= set(current.merge_ids):
            raise ValueError('Removed IDs must belong to the merge session.')
        remaining = tuple(
            cluster_id for cluster_id in current.merge_ids if cluster_id not in cluster_ids
        )
        merge = MergeSession(current.reference_id, remaining, current.merge.entry_snapshot)
        after = CurationSelectionState(
            mode=WorkflowMode.MERGE,
            similar_ids=_ordered_union(current.similar_ids, cluster_ids),
            reference_id=current.reference_id,
            merge=merge,
        )
        return self._apply(after)

    def reorder_merge(self, cluster_ids, insertion):
        """Move staged candidates to an insertion point."""
        self._require_merge_mode()
        cluster_ids = _as_unique_ids(cluster_ids)
        current = self._state
        if current.reference_id in cluster_ids:
            raise ValueError('The merge reference cannot be reordered.')
        if not set(cluster_ids) <= set(current.merge_ids):
            raise ValueError('Reordered IDs must belong to the merge session.')
        remaining = [
            cluster_id for cluster_id in current.merge_ids if cluster_id not in cluster_ids
        ]
        if not 1 <= insertion <= len(remaining):
            raise ValueError('Merge insertion must follow the fixed reference.')
        remaining[insertion:insertion] = cluster_ids
        merge = MergeSession(current.reference_id, tuple(remaining), current.merge.entry_snapshot)
        after = CurationSelectionState(
            mode=WorkflowMode.MERGE,
            similar_ids=current.similar_ids,
            reference_id=current.reference_id,
            merge=merge,
        )
        return self._apply(after)

    def _require_normal_mode(self):
        if self._state.is_merge_mode:
            raise RuntimeError('This operation is unavailable in Merge mode.')

    def _require_merge_mode(self):
        if not self._state.is_merge_mode:
            raise RuntimeError('This operation requires Merge mode.')

    def _apply(self, after):
        before = self._state
        self._state = after
        return SelectionChange.create(before, after)
