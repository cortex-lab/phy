"""Tests for the pure curation selection state."""

from dataclasses import FrozenInstanceError

from pytest import raises

from .._selection import (
    CurationSelectionController,
    CurationSelectionState,
    MergeSession,
    NormalWorkflowSnapshot,
    WorkflowMode,
)


def test_state_derives_unique_effective_and_presentation_ids():
    state = CurationSelectionState(cluster_ids=(3, 1), similar_ids=(1, 2))

    assert state.effective_ids == (3, 1, 2)
    assert state.presentation_order == (3, 1, 2)


def test_state_rejects_invalid_ids_reference_and_presentation():
    with raises(ValueError, match='unique'):
        CurationSelectionState(cluster_ids=(1, 1))
    with raises(ValueError, match='reference'):
        CurationSelectionState(cluster_ids=(1,), reference_id=2)
    with raises(ValueError, match='exactly'):
        CurationSelectionState(cluster_ids=(1,), presentation_order=(2,))
    with raises(ValueError, match='first presentation'):
        CurationSelectionState(
            cluster_ids=(1, 2),
            reference_id=2,
            presentation_order=(1, 2),
        )
    with raises(ValueError, match='requires a merge session'):
        CurationSelectionState(mode=WorkflowMode.MERGE)


def test_state_is_immutable():
    state = CurationSelectionState(cluster_ids=(1,), reference_id=1)

    with raises(FrozenInstanceError):
        state.reference_id = 2


def test_set_cluster_selection_uses_blue_first_id_or_explicit_reference():
    controller = CurationSelectionController()

    change = controller.set_cluster_selection((3, 1))
    assert change.after.reference_id == 3
    assert change.presentation_changed
    assert change.reference_changed

    change = controller.set_cluster_selection((1, 2))
    assert change.after.reference_id == 1
    assert change.after.presentation_order == (1, 2)

    change = controller.set_cluster_selection((1, 2), reference_id=2)
    assert change.after.reference_id == 2
    assert change.after.presentation_order == (2, 1)


def test_set_similarity_and_clear_similarity_selection():
    controller = CurationSelectionController(
        CurationSelectionState(cluster_ids=(1,), reference_id=1)
    )

    change = controller.set_similarity_selection((3, 2))
    assert change.after.effective_ids == (1, 3, 2)
    assert change.after.presentation_order == (1, 3, 2)
    assert change.roles_changed
    assert change.presentation_changed
    assert not change.reference_changed

    change = controller.clear_similarity_selection()
    assert change.after.similar_ids == ()
    assert change.after.presentation_order == (1,)


def test_set_normal_selection_replaces_all_roles_atomically():
    controller = CurationSelectionController()

    change = controller.set_normal_selection((3, 1), (2,), reference_id=1)

    assert change.after.cluster_ids == (3, 1)
    assert change.after.similar_ids == (2,)
    assert change.after.reference_id == 1
    assert change.after.presentation_order == (1, 3, 2)


def test_role_transfers_leave_effective_presentation_unchanged():
    controller = CurationSelectionController(
        CurationSelectionState(
            cluster_ids=(1, 2),
            similar_ids=(3,),
            reference_id=1,
            presentation_order=(1, 2, 3),
        )
    )

    change = controller.transfer_cluster_to_similarity((2,))
    assert change.after.cluster_ids == (1,)
    assert change.after.similar_ids == (3, 2)
    assert change.after.presentation_order == (1, 2, 3)
    assert change.roles_changed
    assert not change.presentation_changed

    change = controller.transfer_similarity_to_cluster((3,))
    assert change.after.cluster_ids == (1, 3)
    assert change.after.similar_ids == (2,)
    assert change.after.presentation_order == (1, 2, 3)
    assert change.roles_changed
    assert not change.presentation_changed


def test_zero_reference_survives_similarity_to_cluster_transfer():
    controller = CurationSelectionController(
        CurationSelectionState(cluster_ids=(0,), similar_ids=(4,), reference_id=0)
    )

    change = controller.transfer_similarity_to_cluster((4,))

    assert change.after.cluster_ids == (0, 4)
    assert change.after.reference_id == 0
    assert change.after.presentation_order == (0, 4)


def test_role_transfer_rejects_ids_not_in_the_source_selection():
    controller = CurationSelectionController(
        CurationSelectionState(cluster_ids=(1,), similar_ids=(2,), reference_id=1)
    )

    with raises(ValueError, match='cluster selection'):
        controller.transfer_cluster_to_similarity((2,))
    with raises(ValueError, match='similarity selection'):
        controller.transfer_similarity_to_cluster((1,))
    with raises(ValueError, match='reference'):
        controller.transfer_cluster_to_similarity((1,))


def test_snapshot_restore_and_noop_change_classification():
    controller = CurationSelectionController(
        CurationSelectionState(cluster_ids=(1,), similar_ids=(2,), reference_id=1)
    )
    snapshot = controller.snapshot()
    controller.set_similarity_selection((3,))

    change = controller.restore(snapshot)
    assert change.before.similar_ids == (3,)
    assert change.after is snapshot
    assert change.presentation_changed

    change = controller.restore(snapshot)
    assert not change.changed
    assert not change.roles_changed
    assert not change.presentation_changed


def test_merge_session_validates_reference_and_state_roles():
    snapshot = NormalWorkflowSnapshot((1,), (), 1, (1,))
    with raises(ValueError, match='first staged'):
        MergeSession(1, (2, 1), snapshot)
    merge = MergeSession(1, (1, 2), snapshot)
    with raises(ValueError, match='Cluster selection'):
        CurationSelectionState(
            mode=WorkflowMode.MERGE,
            cluster_ids=(1,),
            reference_id=1,
            merge=merge,
        )
    with raises(ValueError, match='both staged'):
        CurationSelectionState(
            mode=WorkflowMode.MERGE,
            similar_ids=(2,),
            reference_id=1,
            merge=merge,
        )


def test_enter_and_cancel_merge_mode_restore_exact_entry_selection():
    initial = CurationSelectionState(
        cluster_ids=(3, 1),
        similar_ids=(4, 2),
        reference_id=1,
        presentation_order=(1, 3, 4, 2),
    )
    context = {'cluster_sort': ('id', 'asc')}
    controller = CurationSelectionController(initial)

    change = controller.enter_merge_mode(context)

    assert change.mode_changed
    assert change.roles_changed
    assert not change.presentation_changed
    assert change.after.merge_ids == (1, 3, 4, 2)
    assert change.after.cluster_ids == ()
    assert change.after.similar_ids == ()
    assert set(change.after.effective_ids) == set(initial.effective_ids)
    assert change.after.merge.entry_snapshot.workflow_context is context

    change = controller.cancel_merge_mode()
    assert change.after == initial
    assert change.mode_changed
    assert not change.presentation_changed


def test_enter_merge_mode_requires_cluster_selection():
    controller = CurationSelectionController(CurationSelectionState(similar_ids=(2,)))
    with raises(ValueError, match='Cluster View selection'):
        controller.enter_merge_mode()


def test_merge_candidate_transfer_and_reorder_follow_visible_role_order():
    controller = CurationSelectionController(
        CurationSelectionState(cluster_ids=(1, 2), similar_ids=(3,), reference_id=1)
    )
    controller.enter_merge_mode()

    change = controller.set_similarity_selection((4, 5))
    assert change.after.presentation_order == (1, 2, 3, 4, 5)

    change = controller.add_to_merge((4,))
    assert change.after.merge_ids == (1, 2, 3, 4)
    assert change.after.similar_ids == (5,)
    assert change.after.presentation_order == (1, 2, 3, 4, 5)
    assert not change.presentation_changed

    change = controller.remove_from_merge((2,))
    assert change.after.merge_ids == (1, 3, 4)
    assert change.after.similar_ids == (5, 2)
    assert change.after.presentation_order == (1, 3, 4, 5, 2)
    assert change.presentation_changed

    change = controller.reorder_merge((4,), 1)
    assert change.after.merge_ids == (1, 4, 3)
    assert change.after.presentation_order == (1, 4, 3, 5, 2)
    assert change.presentation_changed


def test_merge_candidate_guards_reference_and_duplicate_membership():
    controller = CurationSelectionController(CurationSelectionState(cluster_ids=(1, 2)))
    controller.enter_merge_mode()

    change = controller.add_to_merge((2,))
    assert not change.changed
    with raises(ValueError, match='reference'):
        controller.remove_from_merge((1,))
    with raises(ValueError, match='reference'):
        controller.reorder_merge((1,), 1)
    with raises(ValueError, match='merge session'):
        controller.remove_from_merge((9,))
