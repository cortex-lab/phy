"""Immutable, UI-independent selection state for curation workflows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from phy.utils.selection import SelectionIntent, SelectionMutation


class WorkflowMode(Enum):
    NORMAL = 'normal'
    MERGE = 'merge'


def _as_unique_ids(cluster_ids) -> tuple[int, ...]:
    cluster_ids = tuple(cluster_ids)
    if len(cluster_ids) != len(set(cluster_ids)):
        raise ValueError('Cluster IDs must be unique.')
    return cluster_ids


def _ordered_union(*lists) -> tuple[int, ...]:
    return tuple(dict.fromkeys(cluster_id for ids in lists for cluster_id in ids))


@dataclass(frozen=True)
class NormalWorkflowSnapshot:
    selection: CurationSelectionState
    workflow_context: object = None

    def __post_init__(self):
        if not isinstance(self.selection, CurationSelectionState):
            raise TypeError('Snapshot selection must be a CurationSelectionState.')
        if self.selection.mode is not WorkflowMode.NORMAL:
            raise ValueError('Normal workflow snapshots require a Normal-mode selection.')


@dataclass(frozen=True)
class MergeSession:
    reference_id: int
    ordered_ids: tuple[int, ...]
    entry_snapshot: NormalWorkflowSnapshot
    proposition_id: str | None = None

    def __post_init__(self):
        ordered = _as_unique_ids(self.ordered_ids)
        if not ordered or ordered[0] != self.reference_id:
            raise ValueError('The merge reference must be the first staged cluster.')
        if self.proposition_id is not None and not self.proposition_id:
            raise ValueError('The merge proposition ID cannot be empty.')
        object.__setattr__(self, 'ordered_ids', ordered)


@dataclass(frozen=True)
class CurationSelectionState:
    """Authoritative roles, rendering order, and palette-slot bindings.

    ``color_slots`` is deliberately not derived from presentation order: a
    ``None`` entry is a released palette slot and a non-active ID is a reserved
    binding.
    """

    mode: WorkflowMode = WorkflowMode.NORMAL
    cluster_ids: tuple[int, ...] = ()
    similar_ids: tuple[int, ...] = ()
    reference_id: int | None = None
    presentation_order: tuple[int, ...] | None = None
    color_slots: tuple[int | None, ...] | None = None
    merge: MergeSession | None = None

    def __post_init__(self):
        clusters = _as_unique_ids(self.cluster_ids)
        similar = _as_unique_ids(self.similar_ids)
        reference = self.reference_id
        merge = self.merge
        if self.mode is WorkflowMode.NORMAL:
            if merge is not None:
                raise ValueError('Normal mode cannot contain a merge session.')
            if reference is None and clusters:
                reference = clusters[0]
            if reference is not None and reference not in clusters:
                raise ValueError('The reference ID must belong to the cluster selection.')
            effective = _ordered_union(clusters, similar)
            primary = clusters
        else:
            if merge is None:
                raise ValueError('Merge mode requires a merge session.')
            if clusters:
                raise ValueError('Cluster selection must be empty in Merge mode.')
            reference = merge.reference_id if reference is None else reference
            if reference != merge.reference_id:
                raise ValueError('The selection and merge references must agree.')
            if set(similar) & set(merge.ordered_ids):
                raise ValueError('A cluster cannot be both staged and selected as similar.')
            effective = _ordered_union(merge.ordered_ids, similar)
            primary = merge.ordered_ids
        default_order = _ordered_union(
            (reference,) if reference is not None else (), primary, similar
        )
        presentation = (
            default_order
            if self.presentation_order is None
            else _as_unique_ids(self.presentation_order)
        )
        if set(presentation) != set(effective):
            raise ValueError('Presentation order must contain exactly the effective IDs.')
        if similar and reference is None:
            raise ValueError('Similarity selection requires a reference ID.')
        if presentation and reference is not None and presentation[0] != reference:
            raise ValueError('The reference ID must occupy the first presentation slot.')
        if self.mode is WorkflowMode.MERGE:
            if presentation[: len(primary)] != primary:
                raise ValueError('Merge presentation must begin with the staged merge order.')
            if set(presentation[len(primary) :]) != set(similar):
                raise ValueError('Merge presentation tail must contain the Similarity selection.')
        slots = presentation if self.color_slots is None else tuple(self.color_slots)
        bindings = tuple(cluster_id for cluster_id in slots if cluster_id is not None)
        if len(bindings) != len(set(bindings)):
            raise ValueError('Color-slot bindings must be unique.')
        if reference is None and bindings:
            raise ValueError('Color slots require a reference ID.')
        if not set(effective) <= set(bindings):
            raise ValueError('Color slots must contain every effective ID.')
        if bindings and reference is not None and (not slots or slots[0] != reference):
            raise ValueError('The reference ID must occupy the first color slot.')
        object.__setattr__(self, 'cluster_ids', clusters)
        object.__setattr__(self, 'similar_ids', similar)
        object.__setattr__(self, 'reference_id', reference)
        object.__setattr__(self, 'presentation_order', presentation)
        object.__setattr__(self, 'color_slots', slots)

    @property
    def color_indices(self):
        """Immutable ``cluster_id -> palette slot`` projection."""
        return MappingProxyType(
            {
                cluster_id: index
                for index, cluster_id in enumerate(self.color_slots)
                if cluster_id is not None
            }
        )

    @property
    def effective_ids(self):
        return _ordered_union(self.merge_ids, self.similar_ids)

    @property
    def merge_ids(self):
        return self.merge.ordered_ids if self.merge is not None else self.cluster_ids

    @property
    def is_merge_mode(self):
        return self.mode is WorkflowMode.MERGE


CurationSelectionSnapshot = CurationSelectionState


@dataclass(frozen=True)
class SelectionChange:
    before: CurationSelectionState
    after: CurationSelectionState
    roles_changed: bool
    presentation_changed: bool
    colors_changed: bool
    reference_changed: bool
    mode_changed: bool

    @classmethod
    def create(cls, before, after):
        return cls(
            before,
            after,
            before.cluster_ids != after.cluster_ids
            or before.similar_ids != after.similar_ids
            or before.merge_ids != after.merge_ids,
            before.presentation_order != after.presentation_order,
            before.color_slots != after.color_slots,
            before.reference_id != after.reference_id,
            before.mode is not after.mode,
        )

    @property
    def changed(self):
        return self.before != self.after

    @property
    def render_changed(self):
        return self.presentation_changed or self.colors_changed


class CurationSelectionController:
    def __init__(self, state=None):
        self._state = state or CurationSelectionState()

    @property
    def state(self):
        return self._state

    def snapshot(self):
        return self._state

    def restore(self, snapshot):
        if not isinstance(snapshot, CurationSelectionState):
            raise TypeError('Expected a CurationSelectionState snapshot.')
        return self._apply(snapshot)

    def set_cluster_selection(self, cluster_ids, reference_id=None):
        self._require_normal_mode()
        clusters = _as_unique_ids(cluster_ids)
        reference = clusters[0] if reference_id is None and clusters else reference_id
        similar = self._state.similar_ids if reference is not None else ()
        order = _ordered_union((reference,) if reference is not None else (), clusters, similar)
        slots = self._slots_for_normal_roles(clusters, similar, reference, order)
        return self._apply(
            CurationSelectionState(
                cluster_ids=clusters,
                similar_ids=similar,
                reference_id=reference,
                presentation_order=order,
                color_slots=slots,
            )
        )

    def set_normal_selection(
        self, cluster_ids, similar_ids=(), reference_id=None, presentation_order=None
    ):
        clusters, similar = _as_unique_ids(cluster_ids), _as_unique_ids(similar_ids)
        reference = clusters[0] if reference_id is None and clusters else reference_id
        order = presentation_order or _ordered_union(
            (reference,) if reference is not None else (), clusters, similar
        )
        slots = self._slots_for_normal_roles(clusters, similar, reference, order)
        return self._apply(
            CurationSelectionState(
                cluster_ids=clusters,
                similar_ids=similar,
                reference_id=reference,
                presentation_order=order,
                color_slots=slots,
            )
        )

    def apply_similarity_mutation(self, mutation, presentation_order=None):
        """Reduce one canonical Similarity operation into a complete state."""
        if not isinstance(mutation, SelectionMutation):
            raise TypeError('Expected a SelectionMutation.')
        current = self._state
        if mutation.before_ids != current.similar_ids:
            raise ValueError('Mutation before_ids do not match the current Similarity selection.')
        similar = mutation.after_ids
        if mutation.intent is SelectionIntent.NAVIGATE and len(similar) > 1:
            raise ValueError('Similarity navigation selects at most one candidate.')
        effective = _ordered_union(current.merge_ids, similar)
        order = presentation_order or _ordered_union(
            tuple(
                cluster_id for cluster_id in current.presentation_order if cluster_id in effective
            ),
            effective,
        )
        if current.is_merge_mode:
            slots = self._merge_slots(effective)
        elif mutation.intent in (
            SelectionIntent.REPLACE,
            SelectionIntent.NAVIGATE,
            SelectionIntent.CLEAR,
        ):
            ordered_similar = tuple(
                cluster_id for cluster_id in order if cluster_id in set(similar)
            )
            slots = self._replace_similarity_slots(ordered_similar)
        else:
            slots = self._preserve_and_allocate(current.color_slots, similar)
        return self._apply(
            CurationSelectionState(
                mode=current.mode,
                cluster_ids=current.cluster_ids,
                similar_ids=similar,
                reference_id=current.reference_id,
                presentation_order=order,
                color_slots=slots,
                merge=current.merge,
            )
        )

    def clear_similarity_selection(self):
        current = self._state
        return self.apply_similarity_mutation(
            SelectionMutation.create(SelectionIntent.CLEAR, current.similar_ids, ())
        )

    def set_presentation_order(self, presentation_order):
        current = self._state
        return self._apply(
            CurationSelectionState(
                mode=current.mode,
                cluster_ids=current.cluster_ids,
                similar_ids=current.similar_ids,
                reference_id=current.reference_id,
                presentation_order=_as_unique_ids(presentation_order),
                color_slots=current.color_slots,
                merge=current.merge,
            )
        )

    def enter_merge_mode(self, workflow_context=None):
        self._require_normal_mode()
        current = self._state
        if not current.cluster_ids:
            raise ValueError('Merge mode requires a Cluster View selection.')
        merge = MergeSession(
            current.reference_id,
            current.presentation_order,
            NormalWorkflowSnapshot(current, workflow_context),
        )
        return self._apply(
            CurationSelectionState(
                mode=WorkflowMode.MERGE,
                reference_id=current.reference_id,
                presentation_order=current.presentation_order,
                color_slots=current.color_slots,
                merge=merge,
            )
        )

    def enter_merge_proposition(self, proposition_id, ordered_ids, workflow_context=None):
        """Stage an external merge proposition without changing its entry snapshot."""
        self._require_normal_mode()
        ordered = _as_unique_ids(ordered_ids)
        if len(ordered) < 2:
            raise ValueError('A merge proposition requires at least two cluster IDs.')
        if not proposition_id:
            raise ValueError('The merge proposition ID cannot be empty.')
        current = self._state
        merge = MergeSession(
            ordered[0],
            ordered,
            NormalWorkflowSnapshot(current, workflow_context),
            proposition_id=str(proposition_id),
        )
        return self._apply(
            CurationSelectionState(
                mode=WorkflowMode.MERGE,
                reference_id=ordered[0],
                presentation_order=ordered,
                color_slots=ordered,
                merge=merge,
            )
        )

    def switch_merge_proposition(self, proposition_id, ordered_ids):
        """Replace the active Merge workspace while preserving its Normal entry snapshot."""
        self._require_merge_mode()
        ordered = _as_unique_ids(ordered_ids)
        if len(ordered) < 2:
            raise ValueError('A merge proposition requires at least two cluster IDs.')
        if not proposition_id:
            raise ValueError('The merge proposition ID cannot be empty.')
        merge = MergeSession(
            ordered[0],
            ordered,
            self._state.merge.entry_snapshot,
            proposition_id=str(proposition_id),
        )
        return self._apply(
            CurationSelectionState(
                mode=WorkflowMode.MERGE,
                reference_id=ordered[0],
                presentation_order=ordered,
                color_slots=ordered,
                merge=merge,
            )
        )

    def cancel_merge_mode(self):
        self._require_merge_mode()
        return self._apply(self._state.merge.entry_snapshot.selection)

    def add_to_merge(self, cluster_ids, insertion=None):
        self._require_merge_mode()
        current = self._state
        requested = _as_unique_ids(cluster_ids)
        new = tuple(cluster_id for cluster_id in requested if cluster_id not in current.merge_ids)
        if not new:
            return self._apply(current)
        ids = list(current.merge_ids)
        insertion = len(ids) if insertion is None else insertion
        if not 1 <= insertion <= len(ids):
            raise ValueError('Merge insertion must follow the fixed reference.')
        ids[insertion:insertion] = new
        merge = MergeSession(
            current.reference_id,
            tuple(ids),
            current.merge.entry_snapshot,
            proposition_id=current.merge.proposition_id,
        )
        similar = tuple(cluster_id for cluster_id in current.similar_ids if cluster_id not in new)
        effective = _ordered_union(merge.ordered_ids, similar)
        return self._apply(
            CurationSelectionState(
                mode=WorkflowMode.MERGE,
                similar_ids=similar,
                reference_id=current.reference_id,
                merge=merge,
                color_slots=self._merge_slots(effective),
            )
        )

    def remove_from_merge(self, cluster_ids):
        self._require_merge_mode()
        current = self._state
        removed = _as_unique_ids(cluster_ids)
        if current.reference_id in removed:
            raise ValueError('The merge reference cannot be removed.')
        if not set(removed) <= set(current.merge_ids):
            raise ValueError('Removed IDs must belong to the merge session.')
        merge = MergeSession(
            current.reference_id,
            tuple(i for i in current.merge_ids if i not in removed),
            current.merge.entry_snapshot,
            proposition_id=current.merge.proposition_id,
        )
        similar = _ordered_union(current.similar_ids, removed)
        effective = _ordered_union(merge.ordered_ids, similar)
        return self._apply(
            CurationSelectionState(
                mode=WorkflowMode.MERGE,
                similar_ids=similar,
                reference_id=current.reference_id,
                merge=merge,
                color_slots=self._merge_slots(effective),
            )
        )

    def deselect_from_merge(self, cluster_ids):
        """Remove staged IDs entirely, promoting the next staged ID to reference."""
        self._require_merge_mode()
        current = self._state
        removed = _as_unique_ids(cluster_ids)
        if not set(removed) <= set(current.merge_ids):
            raise ValueError('Deselected IDs must belong to the merge session.')
        merge_ids = tuple(
            cluster_id for cluster_id in current.merge_ids if cluster_id not in removed
        )
        if not merge_ids:
            raise ValueError('The last staged merge cluster cannot be deselected.')
        reference = merge_ids[0]
        merge = MergeSession(
            reference,
            merge_ids,
            current.merge.entry_snapshot,
            proposition_id=current.merge.proposition_id,
        )
        slots = list(current.color_slots)
        if reference != current.reference_id:
            reference_slot = slots.index(reference)
            slots[0], slots[reference_slot] = slots[reference_slot], slots[0]
        return self._apply(
            CurationSelectionState(
                mode=WorkflowMode.MERGE,
                similar_ids=current.similar_ids,
                reference_id=reference,
                color_slots=tuple(slots),
                merge=merge,
            )
        )

    def reorder_merge(self, cluster_ids, insertion):
        self._require_merge_mode()
        current = self._state
        moving = _as_unique_ids(cluster_ids)
        if current.reference_id in moving:
            raise ValueError('The merge reference cannot be reordered.')
        if not set(moving) <= set(current.merge_ids):
            raise ValueError('Reordered IDs must belong to the merge session.')
        remain = [i for i in current.merge_ids if i not in moving]
        if not 1 <= insertion <= len(remain):
            raise ValueError('Merge insertion must follow the fixed reference.')
        remain[insertion:insertion] = moving
        merge = MergeSession(
            current.reference_id,
            tuple(remain),
            current.merge.entry_snapshot,
            proposition_id=current.merge.proposition_id,
        )
        return self._apply(
            CurationSelectionState(
                mode=WorkflowMode.MERGE,
                similar_ids=current.similar_ids,
                reference_id=current.reference_id,
                merge=merge,
                color_slots=current.color_slots,
            )
        )

    def _slots_for_normal_roles(self, clusters, similar, reference, order):
        current = self._state
        if reference != current.reference_id:
            return tuple(order)
        slots = list(current.color_slots)
        active = set(clusters) | set(similar)
        # Removed Cluster IDs are not Similarity reservations.
        for index, cluster_id in enumerate(slots):
            if cluster_id in current.cluster_ids and cluster_id not in active:
                slots[index] = None
        return self._preserve_and_allocate(
            tuple(slots), _ordered_union(clusters, similar), primary_ids=clusters
        )

    def _replace_similarity_slots(self, similar):
        current = self._state
        primary = set(current.cluster_ids)
        slots = list(current.color_slots)
        for index, cluster_id in enumerate(slots):
            if cluster_id not in primary:
                slots[index] = None
        return self._preserve_and_allocate(tuple(slots), similar, primary_ids=current.cluster_ids)

    def _preserve_and_allocate(self, slots, ids, primary_ids=()):
        slots = list(slots)
        existing = {cluster_id for cluster_id in slots if cluster_id is not None}
        primary = set(primary_ids)
        start = (
            max((i for i, cluster_id in enumerate(slots) if cluster_id in primary), default=-1) + 1
        )
        for cluster_id in ids:
            if cluster_id in existing:
                continue
            hole = next((i for i in range(start, len(slots)) if slots[i] is None), None)
            if hole is None:
                slots.append(cluster_id)
            else:
                slots[hole] = cluster_id
            existing.add(cluster_id)
        return tuple(slots)

    def _merge_slots(self, effective):
        return self._preserve_and_allocate(self._state.color_slots, effective)

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
