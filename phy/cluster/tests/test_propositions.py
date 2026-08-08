"""Tests for the pure merge-proposition domain and codecs."""

from dataclasses import FrozenInstanceError

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
