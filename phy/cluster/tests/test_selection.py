"""Tests for the pure curation selection state."""

from dataclasses import FrozenInstanceError

from pytest import raises

from phy.utils.selection import SelectionIntent, SelectionMutation

from .._selection import (
    CurationSelectionController,
    CurationSelectionState,
    MergeSession,
    NormalWorkflowSnapshot,
    WorkflowMode,
)


def _mutate_similarity(controller, after_ids, intent=SelectionIntent.REPLACE):
    return controller.apply_similarity_mutation(
        SelectionMutation.create(intent, controller.state.similar_ids, after_ids)
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
    with raises(ValueError, match='Color slots'):
        CurationSelectionState(cluster_ids=(1, 2), color_slots=(1,))
    with raises(ValueError, match='first color'):
        CurationSelectionState(cluster_ids=(1, 2), color_slots=(2, 1))
    with raises(ValueError, match='Similarity selection'):
        CurationSelectionState(similar_ids=(2,))
    with raises(ValueError, match='Color slots'):
        CurationSelectionState(color_slots=(2,))


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


def test_empty_cluster_selection_clears_similarity_and_color_session():
    controller = CurationSelectionController(
        CurationSelectionState(cluster_ids=(1,), similar_ids=(2,), reference_id=1)
    )

    change = controller.set_cluster_selection(())

    assert change.after.cluster_ids == ()
    assert change.after.similar_ids == ()
    assert change.after.reference_id is None
    assert change.after.presentation_order == ()
    assert change.after.color_slots == ()


def test_set_similarity_and_clear_similarity_selection():
    controller = CurationSelectionController(
        CurationSelectionState(cluster_ids=(1,), reference_id=1)
    )

    change = _mutate_similarity(controller, (3, 2))
    assert change.after.effective_ids == (1, 3, 2)
    assert change.after.presentation_order == (1, 3, 2)
    assert change.roles_changed
    assert change.presentation_changed
    assert not change.reference_changed

    change = controller.clear_similarity_selection()
    assert change.after.similar_ids == ()
    assert change.after.presentation_order == (1,)


def test_toggle_deselection_reserves_and_reselection_restores_color_slots():
    controller = CurationSelectionController(
        CurationSelectionState(cluster_ids=(1,), reference_id=1)
    )

    controller.apply_similarity_mutation(
        SelectionMutation(SelectionIntent.EXTEND, (), (2, 3, 4), (2, 3, 4), ())
    )
    slots = controller.state.color_slots
    change = controller.apply_similarity_mutation(
        SelectionMutation(SelectionIntent.TOGGLE, (2, 3, 4), (2, 4), (), (3,))
    )

    assert change.after.color_slots == slots
    assert not change.colors_changed
    change = controller.apply_similarity_mutation(
        SelectionMutation(SelectionIntent.TOGGLE, (2, 4), (2, 3, 4), (3,), ())
    )
    assert change.after.color_slots == slots
    assert not change.colors_changed


def test_similarity_navigation_reuses_outgoing_or_inactive_color_slot():
    controller = CurationSelectionController(
        CurationSelectionState(
            cluster_ids=(1,),
            similar_ids=(2,),
            reference_id=1,
            color_slots=(1, 2, 3),
        )
    )

    change = _mutate_similarity(controller, (3,), SelectionIntent.NAVIGATE)
    assert change.after.similar_ids == (3,)
    assert change.after.presentation_order == (1, 3)
    assert change.after.color_slots == (1, 3, None)
    assert change.colors_changed

    change = _mutate_similarity(controller, (2,), SelectionIntent.NAVIGATE)
    assert change.after.color_slots == (1, 2, None)

    controller.clear_similarity_selection()
    change = _mutate_similarity(controller, (4,), SelectionIntent.NAVIGATE)
    assert change.after.color_slots == (1, 4, None)


def test_similarity_navigation_preserves_primary_colors_and_selects_at_most_one():
    controller = CurationSelectionController(
        CurationSelectionState(
            cluster_ids=(1, 4),
            similar_ids=(2,),
            reference_id=1,
            color_slots=(1, 4, 2, 3),
        )
    )

    change = _mutate_similarity(controller, (3,), SelectionIntent.NAVIGATE)
    assert change.after.color_slots == (1, 4, 3, None)
    with raises(ValueError, match='at most one'):
        _mutate_similarity(controller, (2, 3), SelectionIntent.NAVIGATE)


def test_set_normal_selection_replaces_all_roles_atomically():
    controller = CurationSelectionController()

    change = controller.set_normal_selection((3, 1), (2,), reference_id=1)

    assert change.after.cluster_ids == (3, 1)
    assert change.after.similar_ids == (2,)
    assert change.after.reference_id == 1
    assert change.after.presentation_order == (1, 3, 2)


def test_snapshot_restore_and_noop_change_classification():
    controller = CurationSelectionController(
        CurationSelectionState(cluster_ids=(1,), similar_ids=(2,), reference_id=1)
    )
    snapshot = controller.snapshot()
    _mutate_similarity(controller, (3,))

    change = controller.restore(snapshot)
    assert change.before.similar_ids == (3,)
    assert change.after is snapshot
    assert change.presentation_changed

    change = controller.restore(snapshot)
    assert not change.changed
    assert not change.roles_changed
    assert not change.presentation_changed


def test_merge_session_validates_reference_and_state_roles():
    snapshot = NormalWorkflowSnapshot(CurationSelectionState(cluster_ids=(1,)))
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
    assert change.after.merge.entry_snapshot.selection is initial

    change = controller.cancel_merge_mode()
    assert change.after == initial
    assert change.mode_changed
    assert not change.presentation_changed


def test_normal_workflow_snapshot_requires_a_normal_selection_state():
    with raises(TypeError, match='CurationSelectionState'):
        NormalWorkflowSnapshot((1,))

    controller = CurationSelectionController(CurationSelectionState(cluster_ids=(1,)))
    controller.enter_merge_mode()
    with raises(ValueError, match='Normal-mode'):
        NormalWorkflowSnapshot(controller.state)


def test_enter_merge_mode_stages_normal_presentation_order():
    initial = CurationSelectionState(
        cluster_ids=(1, 2),
        similar_ids=(3, 4),
        reference_id=1,
        presentation_order=(1, 2, 4, 3),
    )
    controller = CurationSelectionController(initial)

    change = controller.enter_merge_mode()

    assert change.after.merge_ids == initial.presentation_order
    assert not change.presentation_changed


def test_enter_merge_proposition_preserves_entry_and_uses_proposed_reference():
    initial = CurationSelectionState(
        cluster_ids=(3, 1),
        similar_ids=(4,),
        reference_id=3,
        color_slots=(3, 1, 4),
    )
    context = {'cluster_filter': 'group == good'}
    controller = CurationSelectionController(initial)

    change = controller.enter_merge_proposition('merge:8,2', (8, 2), context)

    assert change.after.merge_ids == (8, 2)
    assert change.after.reference_id == 8
    assert change.after.color_slots == (8, 2)
    assert change.after.merge.proposition_id == 'merge:8,2'
    assert change.after.merge.entry_snapshot.selection is initial
    assert change.after.merge.entry_snapshot.workflow_context is context
    assert controller.cancel_merge_mode().after is initial


def test_enter_merge_proposition_validates_identity_and_membership():
    controller = CurationSelectionController(CurationSelectionState(cluster_ids=(1,)))

    with raises(ValueError, match='at least two'):
        controller.enter_merge_proposition('p', (1,))
    with raises(ValueError, match='cannot be empty'):
        controller.enter_merge_proposition('', (1, 2))
    with raises(ValueError, match='unique'):
        controller.enter_merge_proposition('p', (1, 1))


def test_switch_merge_proposition_preserves_original_normal_entry_snapshot():
    initial = CurationSelectionState(cluster_ids=(1,), similar_ids=(2,))
    context = {'cluster_filter': 'group == good'}
    controller = CurationSelectionController(initial)
    controller.enter_merge_mode(context)

    change = controller.switch_merge_proposition('p1', (8, 3))

    assert change.after.merge_ids == (8, 3)
    assert change.after.reference_id == 8
    assert change.after.similar_ids == ()
    assert change.after.merge.proposition_id == 'p1'
    assert change.after.merge.entry_snapshot.selection is initial
    assert change.after.merge.entry_snapshot.workflow_context is context

    first_snapshot = change.after.merge.entry_snapshot
    change = controller.switch_merge_proposition('p2', (7, 4))

    assert change.after.merge_ids == (7, 4)
    assert change.after.merge.entry_snapshot is first_snapshot
    assert controller.cancel_merge_mode().after is initial


def test_switch_merge_proposition_requires_merge_mode_and_valid_proposition():
    controller = CurationSelectionController(CurationSelectionState(cluster_ids=(1,)))

    with raises(RuntimeError, match='requires Merge mode'):
        controller.switch_merge_proposition('p', (1, 2))

    controller.enter_merge_mode()
    with raises(ValueError, match='at least two'):
        controller.switch_merge_proposition('p', (1,))
    with raises(ValueError, match='cannot be empty'):
        controller.switch_merge_proposition('', (1, 2))
    with raises(ValueError, match='unique'):
        controller.switch_merge_proposition('p', (1, 1))


def test_merge_workspace_edits_preserve_proposition_identity():
    controller = CurationSelectionController(CurationSelectionState(cluster_ids=(1,)))
    controller.enter_merge_proposition('p', (1, 2))

    controller.add_to_merge((3,))
    assert controller.state.merge.proposition_id == 'p'
    controller.remove_from_merge((2,))
    assert controller.state.merge.proposition_id == 'p'
    controller.reorder_merge((3,), 1)
    assert controller.state.merge.proposition_id == 'p'


def test_enter_merge_mode_requires_cluster_selection():
    controller = CurationSelectionController()
    with raises(ValueError, match='Cluster View selection'):
        controller.enter_merge_mode()


def test_merge_candidate_transfer_and_reorder_follow_visible_role_order():
    controller = CurationSelectionController(
        CurationSelectionState(cluster_ids=(1, 2), similar_ids=(3,), reference_id=1)
    )
    controller.enter_merge_mode()

    change = _mutate_similarity(controller, (4, 5))
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


def test_presentation_order_transition_preserves_roles_and_colors():
    controller = CurationSelectionController(
        CurationSelectionState(
            cluster_ids=(1, 2), similar_ids=(3,), reference_id=1, color_slots=(1, 2, 3)
        )
    )

    change = controller.set_presentation_order((1, 3, 2))

    assert change.presentation_changed
    assert not change.roles_changed
    assert not change.colors_changed
    assert change.after.cluster_ids == (1, 2)
    assert change.after.similar_ids == (3,)
    assert change.after.color_slots == (1, 2, 3)
    with raises(ValueError, match='exactly'):
        controller.set_presentation_order((1, 2))


def test_merge_presentation_order_requires_merge_prefix_and_similarity_tail():
    controller = CurationSelectionController(CurationSelectionState(cluster_ids=(1, 2)))
    controller.enter_merge_mode()
    _mutate_similarity(controller, (3,))

    change = controller.set_presentation_order((1, 2, 3))

    assert not change.roles_changed
    assert not change.colors_changed
    with raises(ValueError, match='begin'):
        controller.set_presentation_order((1, 3, 2))
    with raises(ValueError, match='exactly'):
        controller.set_presentation_order((1, 2, 4))


def test_replace_and_navigate_reuse_first_similarity_slot_after_cluster_roles():
    controller = CurationSelectionController(
        CurationSelectionState(cluster_ids=(1, 8), reference_id=1)
    )

    controller.apply_similarity_mutation(
        SelectionMutation(SelectionIntent.REPLACE, (), (5,), (5,), ())
    )
    assert controller.state.color_slots == (1, 8, 5)
    controller.apply_similarity_mutation(
        SelectionMutation(SelectionIntent.REPLACE, (5,), (7,), (7,), (5,))
    )
    assert controller.state.color_slots == (1, 8, 7)
    controller.apply_similarity_mutation(
        SelectionMutation(SelectionIntent.NAVIGATE, (7,), (6,), (6,), (7,))
    )
    assert controller.state.color_slots == (1, 8, 6)


def test_extend_fills_released_holes_and_exposes_immutable_palette_projection():
    controller = CurationSelectionController(
        CurationSelectionState(cluster_ids=(1,), reference_id=1, color_slots=(1, None, None))
    )

    controller.apply_similarity_mutation(
        SelectionMutation(SelectionIntent.EXTEND, (), (4, 5), (4, 5), ())
    )
    assert controller.state.color_slots == (1, 4, 5)
    assert dict(controller.state.color_indices) == {1: 0, 4: 1, 5: 2}
    with raises(TypeError):
        controller.state.color_indices[4] = 9


def test_clear_releases_similarity_reservations_for_the_next_replace():
    controller = CurationSelectionController(
        CurationSelectionState(cluster_ids=(1,), reference_id=1)
    )
    controller.apply_similarity_mutation(
        SelectionMutation(SelectionIntent.EXTEND, (), (2, 3), (2, 3), ())
    )
    controller.apply_similarity_mutation(
        SelectionMutation(SelectionIntent.TOGGLE, (2, 3), (2,), (), (3,))
    )
    controller.apply_similarity_mutation(
        SelectionMutation(SelectionIntent.CLEAR, (2,), (), (), (2,))
    )
    assert controller.state.color_slots == (1, None, None)
    controller.apply_similarity_mutation(
        SelectionMutation(SelectionIntent.REPLACE, (), (9,), (9,), ())
    )
    assert controller.state.color_slots == (1, 9, None)


def test_merge_freezes_existing_bindings_and_unseen_candidates_fill_holes():
    initial = CurationSelectionState(
        cluster_ids=(1, 2),
        similar_ids=(3,),
        reference_id=1,
        color_slots=(1, 2, 3, None),
    )
    controller = CurationSelectionController(initial)
    controller.enter_merge_mode()
    entry_slots = controller.state.color_slots
    controller.apply_similarity_mutation(
        SelectionMutation(SelectionIntent.REPLACE, (), (4,), (4,), ())
    )
    assert controller.state.color_slots == (1, 2, 3, 4)
    controller.add_to_merge((4,))
    controller.reorder_merge((4,), 1)
    controller.remove_from_merge((2,))
    assert controller.state.color_indices[1] == 0
    assert controller.state.color_indices[2] == 1
    assert controller.state.color_indices[3] == 2
    assert controller.state.color_indices[4] == 3
    controller.cancel_merge_mode()
    assert controller.state == initial
    assert entry_slots == initial.color_slots
