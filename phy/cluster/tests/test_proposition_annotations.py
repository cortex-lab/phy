"""Pending source membership is independent of traces, tables, and lineage aliases."""

from phy.cluster._proposition_annotations import (
    WHITE,
    project_proposition_labels,
    proposition_label_color,
)
from phy.cluster._propositions import PropositionStatus, decode_curation_mapping


def _catalog(groups, live=None):
    units = list(dict.fromkeys(unit for group in groups for unit in group))
    catalog = decode_curation_mapping(
        {
            'format_version': '2',
            'unit_ids': units,
            'merges': [{'unit_ids': group} for group in groups],
        }
    )
    return catalog if live is None else catalog.project_live_ids(live)


def test_pdf_groups_and_neutral_unit_ignore_display_order():
    catalog = _catalog([[449, 461, 466, 467, 469], [479, 491, 498]])
    keys = tuple(p.key for p in catalog.propositions)
    units = [449, 461, 466, 467, 469, 479, 491, 498, 511]
    for ordered in (units, units[::-1]):
        result = project_proposition_labels(catalog, ordered, keys[:1])
        assert [group.display_id for group in result.groups] == ['P1', 'P2']
        assert result.groups[0].active and not result.groups[1].active
        for unit in units[:5]:
            assert result.annotations[unit].color == proposition_label_color(0)
        for unit in units[5:8]:
            assert result.annotations[unit].color == proposition_label_color(1)
        assert result.annotations[511].color == WHITE
        assert not result.annotations[511].groups


def test_multiple_active_groups_overlap_and_paging_reserve_color_slots():
    catalog = _catalog([[1, 2], [2, 3], [4, 5], [6, 7]])
    keys = tuple(p.key for p in catalog.propositions)
    active = (keys[1], keys[0])
    first = project_proposition_labels(catalog, [1, 2, 3, 4], active)
    overlap = first.annotations[2]
    assert overlap.text == '2*'
    assert [group.key for group in overlap.groups] == list(active)
    assert overlap.color == proposition_label_color(0)
    assert [group.active for group in first.groups] == [True, True, False]
    second = project_proposition_labels(catalog, [6], active, first.color_keys)
    assert second.annotations[6].color == proposition_label_color(3)
    third = project_proposition_labels(catalog, [4, 2], active, second.color_keys)
    assert third.annotations[4].color == first.annotations[4].color
    assert third.annotations[2] == overlap
    # A completed group releases its annotation without shifting other groups'
    # reserved colors in this workspace.
    reviewed = catalog.reject(keys[2])
    fourth = project_proposition_labels(reviewed, [4, 6], active, third.color_keys)
    assert fourth.annotations[4].color == WHITE
    assert fourth.annotations[6].color == second.annotations[6].color


def test_only_authoritative_pending_entries_receive_annotations():
    catalog = _catalog([[1, 2], [3, 4], [5, 6], [7, 8]])
    keys = tuple(p.key for p in catalog.propositions)
    catalog = catalog.reject(keys[0]).accept(keys[1], [3, 4], 30)
    catalog = catalog.accept(keys[2], [5, 6, 7], 50)
    catalog = catalog.project_live_ids(
        [1, 2, 3, 4, 5, 6, 7, 80, 30, 50],
        retired_units={
            '8': {'operation': 'threshold_split', 'replacements': [80]},
            '30': {'operation': 'restore_parents', 'replacements': [3, 4]},
        },
    )
    assert [catalog.status_for(key) for key in keys] == [
        PropositionStatus.REJECTED,
        PropositionStatus.ACCEPTED_RESTORED,
        PropositionStatus.ACCEPTED_MODIFIED,
        PropositionStatus.STALE,
    ]
    result = project_proposition_labels(catalog, [1, 3, 5, 7, 80], keys)
    assert not result.groups
    assert all(annotation.color == WHITE for annotation in result.annotations.values())
    # A restored/reassembled accepted result is still reviewed, never pending.
    catalog = catalog.project_live_ids(
        catalog.live_unit_ids,
        reassembled_results={keys[1]: {'result_unit_id': 30, 'modified': False}},
    )
    assert catalog.status_for(keys[1]) is PropositionStatus.ACCEPTED
    assert not list(catalog.pending_entries())
    # Reset is explicit; aliases alone never cause this transition.
    catalog = catalog.reset(keys[0])
    assert project_proposition_labels(catalog, [1]).annotations[1].groups[0].key == keys[0]


def test_invalid_source_and_missing_catalog_are_neutral():
    catalog = decode_curation_mapping(
        {
            'format_version': '2',
            'unit_ids': [1, 2],
            'merges': [{'unit_ids': [1, 99]}, {'unit_ids': [1]}],
        }
    )
    for source in (catalog, None):
        result = project_proposition_labels(source, [1, 99])
        assert not result.groups
        assert all(item.color == WHITE for item in result.annotations.values())
