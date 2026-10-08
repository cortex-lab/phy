"""Filesystem boundary for merge-proposition input and review decisions."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

CURATION_FILENAME = 'curation.json'
REVIEW_FILENAME = 'curation_review.json'


class PropositionSourceChangedError(RuntimeError):
    """Raised when producer-owned proposition input changed during curation."""


@dataclass(frozen=True)
class PropositionDocuments:
    curation: Mapping | None
    curation_sha256: str | None
    review: Mapping | None


def file_sha256(path):
    """Return a SHA-256 digest of the exact bytes at *path*."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_mapping(path, *, missing_ok=False):
    """Read one JSON object, optionally treating a missing file as absent."""
    path = Path(path)
    try:
        with path.open(encoding='utf8') as stream:
            value = json.load(stream)
    except FileNotFoundError:
        if missing_ok:
            return None
        raise
    except json.JSONDecodeError as e:
        raise ValueError(f'Invalid JSON in {path.name}: {e.msg}.') from e
    if not isinstance(value, dict):
        raise ValueError(f'{path.name} must contain a JSON object.')
    return value


def load_proposition_documents(dataset_dir):
    """Load optional dataset-local proposition input and review sidecar."""
    dataset_dir = Path(dataset_dir)
    curation_path = dataset_dir / CURATION_FILENAME
    curation = read_json_mapping(curation_path, missing_ok=True)
    digest = file_sha256(curation_path) if curation is not None else None
    review = read_json_mapping(dataset_dir / REVIEW_FILENAME, missing_ok=True)
    return PropositionDocuments(curation, digest, review)


def write_json_atomic(path, mapping):
    """Durably replace *path* with a deterministic JSON representation."""
    if not isinstance(mapping, Mapping):
        raise TypeError('Atomic JSON output must be a mapping.')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f'.{path.name}.',
        suffix='.tmp',
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf8') as stream:
            json.dump(mapping, stream, indent=2, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:  # pragma: no cover - platform/filesystem dependent
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def write_review_document(dataset_dir, mapping, *, expected_curation_sha256=None):
    """Write the review sidecar unless its producer input changed meanwhile."""
    dataset_dir = Path(dataset_dir)
    source_path = dataset_dir / CURATION_FILENAME
    if expected_curation_sha256 is not None:
        try:
            current = file_sha256(source_path)
        except FileNotFoundError as e:
            raise PropositionSourceChangedError(
                f'{CURATION_FILENAME} disappeared after it was loaded.'
            ) from e
        if current != expected_curation_sha256:
            raise PropositionSourceChangedError(
                f'{CURATION_FILENAME} changed after it was loaded; review state was not written.'
            )
    write_json_atomic(dataset_dir / REVIEW_FILENAME, mapping)
