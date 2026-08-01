"""Tests for the pure curation selection state."""

from dataclasses import FrozenInstanceError

from pytest import raises

from .._selection import (
    CurationSelectionController,
    CurationSelectionState,
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
    with raises(ValueError, match='Normal-mode'):
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
