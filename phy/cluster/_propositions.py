"""Pure, immutable domain objects for AIND merge propositions.

This module deliberately knows nothing about Qt, files, spike arrays, or the
curation workflow.  The controller and persistence layers exchange ordinary
JSON-compatible mappings through the codec functions below.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
from types import MappingProxyType

from ._history import History

CURATION_FORMAT_VERSION = '2'
REVIEW_FORMAT_VERSION = '1'


def _ids(values, *, name='unit_ids', minimum=0):
    """Return a validated, ordered tuple of integer IDs."""
    if not isinstance(values, (list, tuple)):
        raise ValueError(f'{name} must be a list of integer IDs.')
    ids = tuple(values)
    if any(not isinstance(unit_id, int) or isinstance(unit_id, bool) for unit_id in ids):
        raise ValueError(f'{name} must contain only integer IDs.')
    if len(ids) < minimum:
        raise ValueError(f'{name} must contain at least {minimum} IDs.')
    if len(ids) != len(set(ids)):
        raise ValueError(f'{name} must contain unique IDs.')
    return ids


def _mapping(value, *, name):
    if not isinstance(value, Mapping):
        raise ValueError(f'{name} must be an object.')
    return value


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def proposition_key(unit_ids) -> str:
    """Return the stable key for one *ordered* source merge proposition."""
    ids = _ids(unit_ids, minimum=2)
    payload = ','.join(str(unit_id) for unit_id in ids).encode('ascii')
    return f'merge:{sha256(payload).hexdigest()[:16]}'


class PropositionStatus(Enum):
    PENDING = 'pending'
    ACCEPTED = 'accepted'
    ACCEPTED_MODIFIED = 'accepted_modified'
    REJECTED = 'rejected'
    INVALID = 'invalid'
    STALE = 'stale'


class ReviewDecision(Enum):
    ACCEPTED = 'accepted'
    REJECTED = 'rejected'


@dataclass(frozen=True)
class MergeProposition:
    """A valid AIND ``merges`` item, preserving its producer provenance."""

    unit_ids: tuple[int, ...]
    new_unit_id: int | None = None
    key: str = field(init=False)

    def __post_init__(self):
        ids = _ids(self.unit_ids, minimum=2)
        if self.new_unit_id is not None and (
            not isinstance(self.new_unit_id, int) or isinstance(self.new_unit_id, bool)
        ):
            raise ValueError('new_unit_id must be an integer when supplied.')
        object.__setattr__(self, 'unit_ids', ids)
        object.__setattr__(self, 'key', proposition_key(ids))

    @property
    def reference_id(self):
        return self.unit_ids[0]

    def source_mapping(self):
        result = {'unit_ids': list(self.unit_ids)}
        if self.new_unit_id is not None:
            result['new_unit_id'] = self.new_unit_id
        return result


@dataclass(frozen=True)
class PropositionReview:
    """A durable curator decision.  Pending is represented by no record."""

    decision: ReviewDecision
    applied_unit_ids: tuple[int, ...] | None = None
    result_unit_id: int | None = None

    def __post_init__(self):
        decision = self.decision
        if not isinstance(decision, ReviewDecision):
            try:
                decision = ReviewDecision(decision)
            except ValueError as e:
                raise ValueError('Unknown proposition review decision.') from e
        if decision is ReviewDecision.ACCEPTED:
            if self.applied_unit_ids is None:
                raise ValueError('Accepted reviews require applied_unit_ids.')
            applied = _ids(self.applied_unit_ids, name='applied_unit_ids', minimum=2)
            if not isinstance(self.result_unit_id, int) or isinstance(self.result_unit_id, bool):
                raise ValueError('Accepted reviews require an integer result_unit_id.')
        else:
            if self.applied_unit_ids is not None or self.result_unit_id is not None:
                raise ValueError('Rejected reviews cannot contain applied merge results.')
            applied = None
        object.__setattr__(self, 'decision', decision)
        object.__setattr__(self, 'applied_unit_ids', applied)

    def mapping(self):
        result = {'decision': self.decision.value}
        if self.decision is ReviewDecision.ACCEPTED:
            result['applied_unit_ids'] = list(self.applied_unit_ids)
            result['result_unit_id'] = self.result_unit_id
        return result


@dataclass(frozen=True)
class PropositionEntry:
    """One source list item, including individually invalid source entries."""

    index: int
    proposition: MergeProposition | None = None
    invalid_reason: str | None = None
    raw_mapping: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if self.index < 0:
            raise ValueError('Entry index must be non-negative.')
        if (self.proposition is None) == (self.invalid_reason is None):
            raise ValueError('An entry must be either valid or invalid.')
        object.__setattr__(
            self, 'raw_mapping', _freeze(_mapping(self.raw_mapping, name='merge entry'))
        )

    @property
    def key(self):
        return self.proposition.key if self.proposition is not None else None


@dataclass(frozen=True)
class MergePropositionCatalog:
    """Immutable source propositions, durable reviews, and live-ID projection."""

    source_unit_ids: tuple[int, ...]
    entries: tuple[PropositionEntry, ...]
    reviews: Mapping[str, PropositionReview] = field(default_factory=dict)
    live_unit_ids: tuple[int, ...] | None = None
    source_mapping: Mapping = field(default_factory=dict)

    def __post_init__(self):
        source_ids = _ids(self.source_unit_ids, minimum=0)
        entries = tuple(self.entries)
        if any(not isinstance(entry, PropositionEntry) for entry in entries):
            raise TypeError('entries must contain PropositionEntry instances.')
        keys = tuple(entry.key for entry in entries if entry.key is not None)
        if len(keys) != len(set(keys)):
            raise ValueError('Duplicate merge propositions are invalid.')
        reviews = dict(self.reviews)
        if any(
            not isinstance(key, str) or not isinstance(value, PropositionReview)
            for key, value in reviews.items()
        ):
            raise TypeError('reviews must map proposition keys to PropositionReview values.')
        live_ids = (
            source_ids
            if self.live_unit_ids is None
            else _ids(self.live_unit_ids, name='live_unit_ids')
        )
        source_mapping = _freeze(_mapping(self.source_mapping, name='curation document'))
        object.__setattr__(self, 'source_unit_ids', source_ids)
        object.__setattr__(self, 'entries', entries)
        object.__setattr__(self, 'reviews', MappingProxyType(reviews))
        object.__setattr__(self, 'live_unit_ids', live_ids)
        object.__setattr__(self, 'source_mapping', source_mapping)

    @property
    def propositions(self):
        return tuple(entry.proposition for entry in self.entries if entry.proposition is not None)

    @property
    def orphaned_reviews(self):
        keys = {proposition.key for proposition in self.propositions}
        return MappingProxyType(
            {key: value for key, value in self.reviews.items() if key not in keys}
        )

    def entry_for(self, key):
        return next((entry for entry in self.entries if entry.key == key), None)

    def status_for(self, key):
        entry = self.entry_for(key)
        if entry is None:
            raise KeyError(key)
        if entry.invalid_reason is not None:
            return PropositionStatus.INVALID
        proposition = entry.proposition
        if not set(proposition.unit_ids) <= set(self.source_unit_ids):
            return PropositionStatus.INVALID
        review = self.reviews.get(key)
        if review is not None:
            if review.decision is ReviewDecision.REJECTED:
                return PropositionStatus.REJECTED
            return (
                PropositionStatus.ACCEPTED
                if review.applied_unit_ids == proposition.unit_ids
                else PropositionStatus.ACCEPTED_MODIFIED
            )
        if not set(proposition.unit_ids) <= set(self.live_unit_ids):
            return PropositionStatus.STALE
        return PropositionStatus.PENDING

    def reason_for(self, key):
        entry = self.entry_for(key)
        if entry is None:
            raise KeyError(key)
        if entry.invalid_reason is not None:
            return entry.invalid_reason
        if self.status_for(key) is PropositionStatus.INVALID:
            return 'One or more unit_ids are absent from curation.json unit_ids.'
        if self.status_for(key) is PropositionStatus.STALE:
            missing = tuple(
                unit_id
                for unit_id in entry.proposition.unit_ids
                if unit_id not in self.live_unit_ids
            )
            return f'One or more source clusters no longer exist: {missing}.'
        return None

    def project_live_ids(self, live_unit_ids):
        return self._replace(live_unit_ids=_ids(live_unit_ids, name='live_unit_ids'))

    def snapshot(self):
        """Return an immutable snapshot suitable for history entries."""
        return self

    def restore(self, snapshot):
        if not isinstance(snapshot, MergePropositionCatalog):
            raise TypeError('Expected a MergePropositionCatalog snapshot.')
        return snapshot

    def reject(self, key):
        self._require_reviewable(key)
        return self._with_review(key, PropositionReview(ReviewDecision.REJECTED))

    def accept(self, key, applied_unit_ids, result_unit_id):
        self._require_reviewable(key)
        return self._with_review(
            key,
            PropositionReview(ReviewDecision.ACCEPTED, applied_unit_ids, result_unit_id),
        )

    def reset(self, key):
        self._require_reviewable(key)
        proposition = self.entry_for(key).proposition
        if not set(proposition.unit_ids) <= set(self.live_unit_ids):
            raise ValueError('Cannot reset a review whose source clusters no longer exist.')
        reviews = dict(self.reviews)
        reviews.pop(key, None)
        return self._replace(reviews=reviews)

    def review_mapping(self, *, source_filename='curation.json', source_sha256=None):
        source = {'filename': source_filename}
        if source_sha256 is not None:
            source['sha256'] = source_sha256
        return {
            'format_version': REVIEW_FORMAT_VERSION,
            'source': source,
            'reviews': {key: review.mapping() for key, review in self.reviews.items()},
        }

    def _require_reviewable(self, key):
        status = self.status_for(key)
        if status in {PropositionStatus.INVALID, PropositionStatus.STALE}:
            raise ValueError(f'Cannot review a {status.value} merge proposition.')

    def _with_review(self, key, review):
        if self.entry_for(key) is None:
            raise KeyError(key)
        reviews = dict(self.reviews)
        reviews[key] = review
        return self._replace(reviews=reviews)

    def _replace(self, **changes):
        values = {
            'source_unit_ids': self.source_unit_ids,
            'entries': self.entries,
            'reviews': self.reviews,
            'live_unit_ids': self.live_unit_ids,
            'source_mapping': self.source_mapping,
        }
        values.update(changes)
        return type(self)(**values)


class MergePropositionController:
    """Mutable history participant around an immutable proposition catalog.

    Only durable review changes are put on this local history stack.  Projecting
    live cluster IDs changes derived stale status, but is neither an unsaved
    curator decision nor an independently undoable action: clustering history
    supplies that context during a global undo/redo.
    """

    def __init__(self, catalog):
        if not isinstance(catalog, MergePropositionCatalog):
            raise TypeError('Expected a MergePropositionCatalog.')
        self._catalog = catalog
        self._history = History(catalog)
        self._saved_reviews = self._review_fingerprint(catalog)

    @property
    def catalog(self):
        return self._catalog

    def snapshot(self):
        return self._catalog.snapshot()

    def restore(self, snapshot):
        self._catalog = self._require_catalog(snapshot)
        return None

    def project_live_ids(self, live_unit_ids):
        self._catalog = self._catalog.project_live_ids(tuple(map(int, live_unit_ids)))
        return None

    def reject(self, key):
        return self._transition(self._catalog.reject(key))

    def accept(self, key, applied_unit_ids, result_unit_id):
        return self._transition(self._catalog.accept(key, applied_unit_ids, result_unit_id))

    def reset(self, key):
        return self._transition(self._catalog.reset(key))

    def undo(self):
        if self._history.undo() is not None:
            self._catalog = self._history.current_item.project_live_ids(
                self._catalog.live_unit_ids
            )
        return None

    def redo(self):
        catalog = self._history.redo()
        if catalog is not None:
            self._catalog = catalog.project_live_ids(self._catalog.live_unit_ids)
        return None

    def mark_saved(self):
        self._saved_reviews = self._review_fingerprint(self._catalog)

    def is_dirty(self):
        return self._review_fingerprint(self._catalog) != self._saved_reviews

    def _transition(self, after):
        if after == self._catalog:
            return None
        self._catalog = after
        self._history.add(after)
        return None

    @staticmethod
    def _require_catalog(snapshot):
        if not isinstance(snapshot, MergePropositionCatalog):
            raise TypeError('Expected a MergePropositionCatalog snapshot.')
        return snapshot

    @staticmethod
    def _review_fingerprint(catalog):
        return tuple(
            sorted(
                (key, review.decision.value, review.applied_unit_ids, review.result_unit_id)
                for key, review in catalog.reviews.items()
            )
        )


def decode_curation_mapping(mapping) -> MergePropositionCatalog:
    """Decode a supported AIND/SpikeInterface v2 curation document.

    Invalid individual ``merges`` entries become visible invalid catalog entries;
    an invalid document shell raises ``ValueError`` so callers can disable only
    the proposition feature while leaving ordinary curation available.
    """
    mapping = _mapping(mapping, name='curation document')
    if str(mapping.get('format_version')) != CURATION_FORMAT_VERSION:
        raise ValueError(
            f'Unsupported curation format version: {mapping.get("format_version")!r}.'
        )
    source_ids = _ids(mapping.get('unit_ids'), name='unit_ids')
    merges = mapping.get('merges', ())
    if not isinstance(merges, list):
        raise ValueError('merges must be a list.')
    entries = []
    seen_keys = set()
    for index, item in enumerate(merges):
        raw = item if isinstance(item, Mapping) else {}
        try:
            item = _mapping(item, name=f'merges[{index}]')
            # Unknown per-merge data is not part of the v2 contract; retain it
            # in source_mapping but do not make a valid proposition ambiguous.
            proposition = MergeProposition(item.get('unit_ids'), item.get('new_unit_id'))
            if proposition.key in seen_keys:
                raise ValueError('Duplicate merge proposition.')
            seen_keys.add(proposition.key)
            entries.append(PropositionEntry(index, proposition=proposition, raw_mapping=raw))
        except (TypeError, ValueError) as e:
            entries.append(PropositionEntry(index, invalid_reason=str(e), raw_mapping=raw))
    return MergePropositionCatalog(source_ids, tuple(entries), source_mapping=mapping)


def encode_curation_mapping(catalog: MergePropositionCatalog):
    """Return the untouched producer-owned source document as plain JSON data."""
    if not isinstance(catalog, MergePropositionCatalog):
        raise TypeError('Expected a MergePropositionCatalog.')
    return _thaw(catalog.source_mapping)


def decode_review_mapping(mapping):
    """Decode a ``curation_review.json`` mapping into durable review records.

    The source descriptor is returned as a plain mapping alongside the records,
    allowing the filesystem adapter to compare its source hash without any I/O
    hidden in this module.
    """
    mapping = _mapping(mapping, name='review document')
    if str(mapping.get('format_version')) != REVIEW_FORMAT_VERSION:
        raise ValueError(f'Unsupported review format version: {mapping.get("format_version")!r}.')
    source = _mapping(mapping.get('source'), name='review source')
    filename = source.get('filename')
    if not isinstance(filename, str) or not filename:
        raise ValueError('review source filename must be a non-empty string.')
    if 'sha256' in source and (not isinstance(source['sha256'], str) or not source['sha256']):
        raise ValueError('review source sha256 must be a non-empty string.')
    reviews_mapping = _mapping(mapping.get('reviews'), name='reviews')
    reviews = {}
    for key, value in reviews_mapping.items():
        if not isinstance(key, str) or not key.startswith('merge:'):
            raise ValueError('Review keys must be merge proposition keys.')
        value = _mapping(value, name=f'review {key}')
        reviews[key] = PropositionReview(
            value.get('decision'),
            value.get('applied_unit_ids'),
            value.get('result_unit_id'),
        )
    return MappingProxyType(dict(source)), MappingProxyType(reviews)


def encode_review_mapping(reviews, *, source_filename='curation.json', source_sha256=None):
    """Serialize review records without requiring a catalog instance."""
    catalog = MergePropositionCatalog((), (), reviews=reviews)
    return catalog.review_mapping(
        source_filename=source_filename,
        source_sha256=source_sha256,
    )
