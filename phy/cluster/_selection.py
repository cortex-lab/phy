"""Immutable selection state used by curation workflows.

This module intentionally has no Qt or Supervisor dependencies.  Views can use
the controller as a synchronous source of selection state, while deciding
separately how and when to render a :class:`SelectionChange`.
"""

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
class CurationSelectionState:
    """The authoritative, immutable Normal-mode curation selection.

    ``presentation_order`` is the effective selection in the order delivered
    to scientific views.  It is independent from the two role-specific orders
    so a future role transfer can leave colors and redraw state untouched.
    """

    mode: WorkflowMode = WorkflowMode.NORMAL
    cluster_ids: tuple[int, ...] = ()
    similar_ids: tuple[int, ...] = ()
    reference_id: int | None = None
    presentation_order: tuple[int, ...] | None = None

    def __post_init__(self):
        if self.mode is not WorkflowMode.NORMAL:
            raise ValueError('Only Normal-mode selection state is supported.')

        cluster_ids = _as_unique_ids(self.cluster_ids)
        similar_ids = _as_unique_ids(self.similar_ids)
        reference_id = self.reference_id
        if reference_id is None and cluster_ids:
            reference_id = cluster_ids[0]
        if reference_id is not None and reference_id not in cluster_ids:
            raise ValueError('The reference ID must belong to the cluster selection.')
        effective_ids = _ordered_union(cluster_ids, similar_ids)
        default_presentation = _ordered_union(
            (reference_id,) if reference_id is not None else (),
            cluster_ids,
            similar_ids,
        )
        presentation_order = (
            default_presentation
            if self.presentation_order is None
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
        """The ordered unique union of Cluster and Similarity selections."""
        return _ordered_union(self.cluster_ids, self.similar_ids)


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
                before.cluster_ids != after.cluster_ids or before.similar_ids != after.similar_ids
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
    """Apply validated, atomic Normal-mode selection transitions."""

    def __init__(self, state=None):
        self._state = state or CurationSelectionState()
        if self._state.mode is not WorkflowMode.NORMAL:
            raise ValueError('Only Normal-mode selection is supported.')

    @property
    def state(self):
        """Return the current immutable selection state."""
        return self._state

    def snapshot(self):
        """Return an immutable snapshot of the current Normal-mode state."""
        return self._state

    def restore(self, snapshot):
        """Restore a previously captured Normal-mode *snapshot*."""
        if not isinstance(snapshot, CurationSelectionState):
            raise TypeError('Expected a CurationSelectionState snapshot.')
        return self._apply(snapshot)

    def set_cluster_selection(self, cluster_ids, reference_id=None):
        """Set Cluster View IDs, using the first (blue) ID as the default reference."""
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
        after = CurationSelectionState(
            cluster_ids=self._state.cluster_ids,
            similar_ids=_as_unique_ids(similar_ids),
            reference_id=self._state.reference_id,
        )
        return self._apply(after)

    def clear_similarity_selection(self):
        """Clear only the Similarity View selection."""
        return self.set_similarity_selection(())

    def transfer_cluster_to_similarity(self, cluster_ids):
        """Move Cluster View IDs to Similarity View without changing presentation."""
        cluster_ids = _as_unique_ids(cluster_ids)
        source_ids = set(cluster_ids)
        current = self._state
        if not source_ids <= set(current.cluster_ids):
            raise ValueError('Transferred IDs must belong to the cluster selection.')
        if current.reference_id in source_ids:
            raise ValueError('The reference ID cannot move to the similarity selection.')
        remaining_clusters = tuple(i for i in current.cluster_ids if i not in source_ids)
        similar_ids = _ordered_union(current.similar_ids, cluster_ids)
        reference_id = (
            current.reference_id
            if current.reference_id in remaining_clusters
            else (remaining_clusters[-1] if remaining_clusters else None)
        )
        after = CurationSelectionState(
            cluster_ids=remaining_clusters,
            similar_ids=similar_ids,
            reference_id=reference_id,
            presentation_order=current.presentation_order,
        )
        return self._apply(after)

    def transfer_similarity_to_cluster(self, cluster_ids):
        """Move Similarity View IDs to Cluster View without changing presentation."""
        cluster_ids = _as_unique_ids(cluster_ids)
        source_ids = set(cluster_ids)
        current = self._state
        if not source_ids <= set(current.similar_ids):
            raise ValueError('Transferred IDs must belong to the similarity selection.')
        similar_ids = tuple(i for i in current.similar_ids if i not in source_ids)
        cluster_selection = _ordered_union(current.cluster_ids, cluster_ids)
        reference_id = current.reference_id or (cluster_ids[-1] if cluster_ids else None)
        after = CurationSelectionState(
            cluster_ids=cluster_selection,
            similar_ids=similar_ids,
            reference_id=reference_id,
            presentation_order=current.presentation_order,
        )
        return self._apply(after)

    def _apply(self, after):
        before = self._state
        self._state = after
        return SelectionChange.create(before, after)
