"""Tests for the pure merge-proposition domain and codecs."""

from dataclasses import FrozenInstanceError

import numpy as np
from pytest import raises

from .._propositions import (
    MergeProposition,
    MergePropositionCatalog,
    MergePropositionController,
    PropositionEntry,
    PropositionReview,
    PropositionStatus,
    ReviewDecision,
    decode_curation_mapping,
    decode_review_mapping,
    encode_curation_mapping,
    encode_review_mapping,
    proposition_key,
)


def _source(merges=None):
    return {
        'format_version': '2',
        'unit_ids': [41, 56, 72],
        'merges': merges if merges is not None else [{'unit_ids': [41, 56]}],
        'manual_labels': {'41': ['good']},
    }


def test_retired_units_explain_stale_without_retargeting_or_overwriting_reviews():
    source = _source([{'unit_ids': [41, 56]}, {'unit_ids': [56, 72]}])
    catalog = decode_curation_mapping(source)
    accepted, pending = [entry.key for entry in catalog.entries]
    catalog = catalog.accept(accepted, [41, 56], 100)
    retired = {
        '41': {'operation': 'merge', 'replacements': [100]},
        '56': {'operation': 'merge', 'replacements': [100]},
        '72': {'operation': 'threshold_split', 'replacements': [101, 102]},
    }
    catalog = catalog.project_live_ids([100, 101, 102], retired_units=retired)
    assert catalog.status_for(accepted) is PropositionStatus.ACCEPTED
    assert catalog.reason_for(accepted) is None
    assert catalog.status_for(pending) is PropositionStatus.STALE
    assert '56 was merged' in catalog.reason_for(pending)
    assert '72 was split' in catalog.reason_for(pending)
    assert encode_curation_mapping(catalog) == source
    retired['56']['operation'] = 'threshold_split'
    assert '56 was merged' in catalog.reason_for(pending)
    controller = MergePropositionController(catalog)
    controller.project_live_ids([100, 101, 102])
    assert controller.catalog.reason_for(pending) == catalog.reason_for(pending)


def test_retired_unit_validation():
    for retired in (
        [],
        False,
        {'01': {'operation': 'merge', 'replacements': [100]}},
        {'41': {'operation': 'label', 'replacements': [100]}},
        {'41': {'operation': 'merge', 'replacements': [41]}},
        {'41': {'operation': 'merge', 'replacements': []}},
        {'41': {'operation': 'threshold_split', 'replacements': [-1]}},
    ):
        with raises(ValueError):
            decode_curation_mapping(_source(), retired_units=retired)


def test_restored_accepted_result_preserves_historical_review_without_live_acceptance():
    catalog = decode_curation_mapping(_source())
    key = catalog.propositions[0].key
    catalog = catalog.accept(key, [41, 56], 100)
    review = catalog.review_mapping()
    restored = catalog.project_live_ids(
        [101, 102, 72],
        retired_units={
            '41': {'operation': 'merge', 'replacements': [100]},
            '56': {'operation': 'merge', 'replacements': [100]},
            '100': {'operation': 'restore_parents', 'replacements': [101, 102]},
        },
    )
    assert restored.status_for(key) is PropositionStatus.ACCEPTED_RESTORED
    assert 'subsequently restored' in restored.reason_for(key)
    assert restored.review_mapping() == review
    assert catalog.status_for(key) is PropositionStatus.ACCEPTED
    with raises(ValueError, match='accepted_restored'):
        restored.accept(key, [101, 102], 103)


def test_nested_result_retirement_also_marks_acceptance_historical():
    catalog = decode_curation_mapping(_source())
    key = catalog.propositions[0].key
    catalog = catalog.accept(key, [41, 56], 100)
    projected = catalog.project_live_ids(
        [103, 104],
        retired_units={
            '100': {'operation': 'merge', 'replacements': [102]},
            '102': {'operation': 'restore_parents', 'replacements': [103, 104]},
        },
    )
    assert projected.status_for(key) is PropositionStatus.ACCEPTED_RESTORED


def _restored_catalog(*, modified=False):
    catalog = decode_curation_mapping(_source())
    key = catalog.propositions[0].key
    applied = [41, 56, 72] if modified else [41, 56]
    catalog = catalog.accept(key, applied, 100)
    retired = {
        '41': {'operation': 'merge', 'replacements': [100]},
        '56': {'operation': 'merge', 'replacements': [100]},
        '100': {'operation': 'restore_parents', 'replacements': [101, 102]},
        '101': {'operation': 'merge', 'replacements': [103]},
        '102': {'operation': 'merge', 'replacements': [103]},
    }
    return catalog.project_live_ids([103, 72], retired_units=retired), key


def test_proven_complete_reassembly_recovers_acceptance_without_retargeting_review():
    restored, key = _restored_catalog()
    historical_review = restored.review_mapping()
    # Retirement convergence alone is insufficient evidence: a restore can
    # distribute different historical memberships among its replacement IDs.
    assert restored.status_for(key) is PropositionStatus.ACCEPTED_RESTORED
    proof = {key: {'result_unit_id': np.int64(103), 'modified': False}}
    reassembled = restored.project_live_ids([103, 72], reassembled_results=proof)
    assert reassembled.status_for(key) is PropositionStatus.ACCEPTED
    assert reassembled.reason_for(key) is None
    assert reassembled.review_mapping() == historical_review
    assert reassembled.reviews[key].result_unit_id == 100
    assert reassembled.reviews[key].applied_unit_ids == (41, 56)
    assert encode_curation_mapping(reassembled) == _source()
    proof[key]['modified'] = True
    assert reassembled.status_for(key) is PropositionStatus.ACCEPTED
    assert type(reassembled.reassembled_results[key]['result_unit_id']) is int
    with raises(TypeError):
        reassembled.reassembled_results[key]['modified'] = True


def test_complete_reassembly_with_extra_membership_is_modified_acceptance():
    restored, key = _restored_catalog()
    reassembled = restored.project_live_ids(
        [103, 72],
        reassembled_results={key: {'result_unit_id': 103, 'modified': True}},
    )
    assert reassembled.status_for(key) is PropositionStatus.ACCEPTED_MODIFIED
    assert reassembled.review_mapping() == restored.review_mapping()
    # Reconstructing an historically modified accepted scope does not convert
    # that curator decision into an exact acceptance of the original proposal.
    restored, key = _restored_catalog(modified=True)
    reassembled = restored.project_live_ids(
        [103, 72],
        reassembled_results={key: {'result_unit_id': 103, 'modified': False}},
    )
    assert reassembled.status_for(key) is PropositionStatus.ACCEPTED_MODIFIED
    assert reassembled.reviews[key].applied_unit_ids == (41, 56, 72)


def test_reassembly_projection_clears_after_partial_split_or_another_restore():
    restored, key = _restored_catalog()
    reassembled = restored.project_live_ids(
        [103, 72],
        reassembled_results={key: {'result_unit_id': 103, 'modified': False}},
    )
    # Even a still-live target may have lost part of its accepted membership.
    # Every caller must prove complete coverage again on each projection.
    assert (
        reassembled.project_live_ids([103, 72]).status_for(key)
        is PropositionStatus.ACCEPTED_RESTORED
    )
    retired = dict(reassembled.retired_units)
    retired['103'] = {'operation': 'restore_parents', 'replacements': [104, 105]}
    separated = reassembled.project_live_ids([104, 105, 72], retired_units=retired)
    assert separated.status_for(key) is PropositionStatus.ACCEPTED_RESTORED
    retired['104'] = {'operation': 'merge', 'replacements': [106]}
    retired['105'] = {'operation': 'merge', 'replacements': [106]}
    repeated = separated.project_live_ids(
        [106, 72],
        retired_units=retired,
        reassembled_results={key: {'result_unit_id': 106, 'modified': False}},
    )
    assert repeated.status_for(key) is PropositionStatus.ACCEPTED
    assert repeated.review_mapping() == restored.review_mapping()


def test_reassembly_is_derived_not_saved_or_a_dirty_review_transition():
    restored, key = _restored_catalog()
    controller = MergePropositionController(restored)
    controller.project_live_ids(
        [103, 72],
        reassembled_results={key: {'result_unit_id': 103, 'modified': False}},
    )
    assert controller.catalog.status_for(key) is PropositionStatus.ACCEPTED
    assert not controller.is_dirty()
    snapshot = controller.snapshot()
    controller.project_live_ids([101, 102, 72])
    assert controller.catalog.status_for(key) is PropositionStatus.ACCEPTED_RESTORED
    controller.restore(snapshot)
    assert controller.catalog.status_for(key) is PropositionStatus.ACCEPTED
    source, reviews = decode_review_mapping(controller.catalog.review_mapping())
    loaded = decode_curation_mapping(encode_curation_mapping(controller.catalog))._replace(
        reviews=reviews
    )
    assert loaded.reviews[key].result_unit_id == 100
    assert not loaded.reassembled_results
    loaded = loaded.project_live_ids([103, 72], retired_units=controller.catalog.retired_units)
    assert loaded.status_for(key) is PropositionStatus.ACCEPTED_RESTORED
    assert 'reassembled_results' not in source


def test_reassembly_projection_requires_a_known_accepted_review_and_live_result():
    restored, key = _restored_catalog()
    for value in (
        [],
        {'result_unit_id': 999, 'modified': False},
        {'result_unit_id': True, 'modified': False},
        {'result_unit_id': 103, 'modified': 0},
        {'result_unit_id': 103},
        {'result_unit_id': 103, 'modified': False, 'unknown': 1},
    ):
        with raises(ValueError):
            restored.project_live_ids([103, 72], reassembled_results={key: value})
    for catalog in (
        decode_curation_mapping(_source()),
        decode_curation_mapping(_source()).reject(key),
    ):
        with raises(ValueError, match='accepted review'):
            catalog.project_live_ids(
                [103, 72],
                reassembled_results={key: {'result_unit_id': 103, 'modified': False}},
            )
    with raises(ValueError, match='accepted review'):
        restored.project_live_ids(
            [103, 72],
            reassembled_results={'merge:unknown': {'result_unit_id': 103, 'modified': False}},
        )


def test_explicit_review_change_drops_old_derived_reassembly():
    restored, key = _restored_catalog()
    reassembled = restored.project_live_ids(
        [103, 72],
        reassembled_results={key: {'result_unit_id': 103, 'modified': False}},
    )
    rejected = reassembled.reject(key)
    assert rejected.status_for(key) is PropositionStatus.REJECTED
    assert not rejected.reassembled_results
    accepted_again = reassembled.accept(key, [103, 72], 104)
    assert accepted_again.reviews[key].result_unit_id == 104
    assert not accepted_again.reassembled_results


def test_retirement_projection_undo_and_durable_modified_review():
    catalog = decode_curation_mapping(_source())
    key = catalog.propositions[0].key
    controller = MergePropositionController(catalog)
    before = controller.snapshot()
    retired = {'41': {'operation': 'threshold_split', 'replacements': [100, 101]}}
    controller.project_live_ids([56, 72, 100, 101], retired_units=retired)
    assert '41 was split' in controller.catalog.reason_for(key)
    controller.restore(before)
    assert controller.catalog.status_for(key) is PropositionStatus.PENDING
    modified = catalog.accept(key, [41, 72], 100).project_live_ids(
        [56, 100], retired_units={'41': {'operation': 'merge', 'replacements': [100]}}
    )
    assert modified.status_for(key) is PropositionStatus.ACCEPTED_MODIFIED
    rejected = catalog.reject(key).project_live_ids([56, 100], retired_units=retired)
    assert rejected.status_for(key) is PropositionStatus.REJECTED


def test_aind_v2_decode_preserves_source_and_proposition_order():
    source = _source([{'unit_ids': [56, 41], 'new_unit_id': 1000}])

    catalog = decode_curation_mapping(source)
    proposition = catalog.propositions[0]

    assert proposition.unit_ids == (56, 41)
    assert proposition.reference_id == 56
    assert proposition.new_unit_id == 1000
    assert catalog.status_for(proposition.key) is PropositionStatus.PENDING
    assert encode_curation_mapping(catalog) == source

    # The catalog owns an immutable source snapshot, rather than the caller's
    # mutable producer mapping.
    source['merges'][0]['unit_ids'][0] = 999
    assert encode_curation_mapping(catalog)['merges'][0]['unit_ids'] == [56, 41]


def test_stable_keys_are_ordered_and_propositions_are_immutable():
    key = proposition_key([41, 56])

    assert key == proposition_key((41, 56))
    assert key != proposition_key((56, 41))
    assert key.startswith('merge:')
    assert len(key) == len('merge:') + 16

    proposition = MergeProposition((41, 56))
    with raises(FrozenInstanceError):
        proposition.unit_ids = (56, 41)


def test_invalid_individual_merges_remain_visible_alongside_valid_entries():
    catalog = decode_curation_mapping(
        _source(
            [
                {'unit_ids': [41, 56]},
                {'unit_ids': [41]},
                {'unit_ids': [41, 41]},
                {'unit_ids': ['56', 72]},
                {'new_unit_id': 99},
                {'unit_ids': [41, 56]},
                'not an object',
            ]
        )
    )

    assert len(catalog.entries) == 7
    assert len(catalog.propositions) == 1
    assert catalog.entries[1].invalid_reason == 'unit_ids must contain at least 2 IDs.'
    assert 'unique' in catalog.entries[2].invalid_reason
    assert 'integer' in catalog.entries[3].invalid_reason
    assert 'list' in catalog.entries[4].invalid_reason
    assert catalog.entries[5].invalid_reason == 'Duplicate merge proposition.'
    assert catalog.entries[6].invalid_reason == 'merges[6] must be an object.'


def test_document_shell_errors_disable_only_the_proposition_feature():
    with raises(ValueError, match='Unsupported'):
        decode_curation_mapping({'format_version': '1', 'unit_ids': [], 'merges': []})
    with raises(ValueError, match='unit_ids'):
        decode_curation_mapping({'format_version': '2', 'merges': []})
    with raises(ValueError, match='merges'):
        decode_curation_mapping({'format_version': '2', 'unit_ids': [], 'merges': {}})


def test_source_and_live_id_validity_are_derived_not_persisted():
    catalog = decode_curation_mapping(_source([{'unit_ids': [41, 99]}, {'unit_ids': [41, 56]}]))
    invalid, valid = catalog.entries

    assert catalog.status_for(invalid.key) is PropositionStatus.INVALID
    assert 'absent' in catalog.reason_for(invalid.key)
    assert catalog.status_for(valid.key) is PropositionStatus.PENDING

    stale = catalog.project_live_ids([41, 72])
    assert stale.status_for(valid.key) is PropositionStatus.STALE
    assert '56' in stale.reason_for(valid.key)
    assert catalog.status_for(valid.key) is PropositionStatus.PENDING


def test_review_transitions_are_immutable_and_modified_acceptance_is_derived():
    catalog = decode_curation_mapping(_source())
    key = catalog.propositions[0].key

    rejected = catalog.reject(key)
    assert rejected.status_for(key) is PropositionStatus.REJECTED
    assert catalog.status_for(key) is PropositionStatus.PENDING
    assert rejected.reset(key).status_for(key) is PropositionStatus.PENDING

    accepted = catalog.accept(key, [41, 56], 1001)
    modified = catalog.accept(key, [41, 56, 72], 1001)
    assert accepted.status_for(key) is PropositionStatus.ACCEPTED
    assert modified.status_for(key) is PropositionStatus.ACCEPTED_MODIFIED
    assert accepted.reviews[key].result_unit_id == 1001
    assert accepted.project_live_ids([72]).status_for(key) is PropositionStatus.ACCEPTED

    with raises(ValueError, match='source clusters'):
        accepted.project_live_ids([72]).reset(key)

    snapshot = accepted.snapshot()
    assert modified.restore(snapshot) is snapshot


def test_proposition_domain_normalizes_numpy_integer_ids():
    proposition = MergeProposition((np.int64(41), np.int32(56)), np.int64(1000))
    review = PropositionReview(
        ReviewDecision.ACCEPTED,
        (np.int64(41), np.int32(56)),
        np.int64(1001),
    )

    assert proposition.unit_ids == (41, 56)
    assert proposition.new_unit_id == 1000
    assert review.applied_unit_ids == (41, 56)
    assert review.result_unit_id == 1001
    assert all(type(unit_id) is int for unit_id in review.applied_unit_ids)


def test_invalid_and_stale_entries_cannot_receive_or_reset_reviews():
    catalog = decode_curation_mapping(_source([{'unit_ids': [41, 99]}, {'unit_ids': [41, 56]}]))
    invalid_key = catalog.entries[0].key
    valid_key = catalog.entries[1].key

    with raises(ValueError, match='invalid'):
        catalog.reject(invalid_key)
    stale = catalog.project_live_ids([41])
    with raises(ValueError, match='stale'):
        stale.accept(valid_key, [41, 56], 1001)


def test_review_sidecar_round_trip_preserves_orphans():
    catalog = decode_curation_mapping(_source()).accept(proposition_key([41, 56]), [41, 56], 1001)
    reviews = dict(catalog.reviews)
    reviews['merge:orphaned'] = PropositionReview(ReviewDecision.REJECTED)

    mapping = encode_review_mapping(
        reviews, source_filename='curation.json', source_sha256='a' * 64
    )
    source, loaded = decode_review_mapping(mapping)

    assert source == {'filename': 'curation.json', 'sha256': 'a' * 64}
    assert loaded == reviews
    reloaded = MergePropositionCatalog(
        catalog.source_unit_ids, catalog.entries, loaded, source_mapping=catalog.source_mapping
    )
    assert reloaded.orphaned_reviews == {
        'merge:orphaned': PropositionReview(ReviewDecision.REJECTED)
    }


def test_review_validation_and_catalog_invariants():
    with raises(ValueError, match='applied_unit_ids'):
        PropositionReview(ReviewDecision.ACCEPTED, None, 1)
    with raises(ValueError, match='Rejected'):
        PropositionReview(ReviewDecision.REJECTED, (1, 2), 3)
    with raises(ValueError, match='Duplicate merge'):
        MergePropositionCatalog(
            (1, 2),
            (
                PropositionEntry(0, MergeProposition((1, 2)), raw_mapping={'unit_ids': [1, 2]}),
                PropositionEntry(1, MergeProposition((1, 2)), raw_mapping={'unit_ids': [1, 2]}),
            ),
        )
    with raises(ValueError, match='Review keys'):
        decode_review_mapping(
            {'format_version': '1', 'source': {'filename': 'curation.json'}, 'reviews': {'x': {}}}
        )


def test_controller_has_history_compatible_review_undo_redo_and_dirty_tracking():
    controller = MergePropositionController(decode_curation_mapping(_source()))
    key = controller.catalog.propositions[0].key

    assert not controller.is_dirty()
    assert controller.reject(key) is None
    assert controller.catalog.status_for(key) is PropositionStatus.REJECTED
    assert controller.is_dirty()

    # Live validity is derived context and must not itself create unsaved work.
    controller.mark_saved()
    controller.project_live_ids([41])
    assert controller.catalog.status_for(key) is PropositionStatus.REJECTED
    assert not controller.is_dirty()

    controller.undo()
    assert controller.catalog.status_for(key) is PropositionStatus.STALE
    assert controller.is_dirty()
    controller.project_live_ids([41, 56, 72])
    assert controller.catalog.status_for(key) is PropositionStatus.PENDING
    assert controller.is_dirty()
    controller.redo()
    assert controller.catalog.status_for(key) is PropositionStatus.REJECTED
    assert not controller.is_dirty()

    snapshot = controller.snapshot()
    controller.restore(snapshot)
    assert controller.catalog is snapshot
    with raises(TypeError, match='snapshot'):
        controller.restore(None)
