"""Tests for merge-proposition filesystem persistence."""

import json

from pytest import raises

from .._proposition_io import (
    PropositionSourceChangedError,
    file_sha256,
    load_proposition_documents,
    read_json_mapping,
    write_json_atomic,
    write_review_document,
)


def test_load_optional_proposition_documents_and_exact_hash(tmp_path):
    assert load_proposition_documents(tmp_path).curation is None

    source = b'{"format_version":"2","unit_ids":[],"merges":[]}\n'
    (tmp_path / 'curation.json').write_bytes(source)
    (tmp_path / 'curation_review.json').write_text(
        '{"format_version":"1","source":{},"reviews":{}}', encoding='utf8'
    )

    documents = load_proposition_documents(tmp_path)
    assert documents.curation['format_version'] == '2'
    assert documents.curation_sha256 == file_sha256(tmp_path / 'curation.json')
    assert documents.review['reviews'] == {}


def test_read_json_mapping_reports_malformed_and_non_object_json(tmp_path):
    path = tmp_path / 'bad.json'
    path.write_text('{bad', encoding='utf8')
    with raises(ValueError, match='Invalid JSON in bad.json'):
        read_json_mapping(path)

    path.write_text('[]', encoding='utf8')
    with raises(ValueError, match='JSON object'):
        read_json_mapping(path)


def test_atomic_json_write_replaces_existing_file_without_temp_leak(tmp_path):
    path = tmp_path / 'curation_review.json'
    path.write_text('{"old": true}', encoding='utf8')

    write_json_atomic(path, {'reviews': {'p': {'decision': 'rejected'}}})

    assert json.loads(path.read_text(encoding='utf8'))['reviews']['p']['decision'] == 'rejected'
    assert list(tmp_path.glob('.curation_review.json.*.tmp')) == []


def test_review_write_detects_source_replacement(tmp_path):
    source = tmp_path / 'curation.json'
    source.write_text('{"format_version":"2"}', encoding='utf8')
    expected = file_sha256(source)
    source.write_text('{"format_version":"2", "merges":[]}', encoding='utf8')

    with raises(PropositionSourceChangedError, match='changed'):
        write_review_document(tmp_path, {'reviews': {}}, expected_curation_sha256=expected)
    assert not (tmp_path / 'curation_review.json').exists()


def test_atomic_write_cleans_temporary_file_when_replace_fails(tmp_path, monkeypatch):
    from .. import _proposition_io

    def fail_replace(source, target):
        raise OSError('replace failed')

    monkeypatch.setattr(_proposition_io.os, 'replace', fail_replace)
    with raises(OSError, match='replace failed'):
        write_json_atomic(tmp_path / 'curation_review.json', {'reviews': {}})
    assert list(tmp_path.glob('.curation_review.json.*.tmp')) == []
