"""Disposable incremental state; persisted journals retain independent full validation."""

import copy
from dataclasses import dataclass

import numpy as np

from ._immutable import freeze, retained_bytes, thaw
from .curation import (
    POLICY,
    _integer,
    interpret_events,
    normalize_metadata,
)
from .metadata import complete_schema
from .source import hash_json


@dataclass(frozen=True)
class Node:
    """An immutable historical membership; restore aliases share the original node."""

    unit: int
    index: int
    parts: object
    expression: dict
    count: object

    def __post_init__(self):
        object.__setattr__(self, 'expression', freeze(self.expression))

    def lineage(self):
        return (
            self.unit,
            self.index,
            tuple(p.lineage() for p in self.parts) if self.parts is not None else None,
        )


def leaves(nodes):
    pending = list(reversed(nodes))
    result, seen = [], set()
    while pending:
        node = pending.pop()
        if node.parts is not None:
            pending.extend(reversed(node.parts))
        else:
            reference = (node.unit, node.index)
            if reference in seen:
                raise ValueError('Recorded restoration contains a repeated historical component.')
            seen.add(reference)
            result.append(node)
    return result


@dataclass(frozen=True)
class SemanticState:
    active: dict
    metadata: dict
    next_id: int
    policy: object
    schema: dict
    membership_key: object

    def __post_init__(self):
        object.__setattr__(self, 'active', freeze(self.active))
        object.__setattr__(self, 'metadata', freeze(self.metadata))
        object.__setattr__(self, 'schema', freeze(self.schema))

    @classmethod
    def initial(cls, session):
        nodes = {
            unit: Node(unit, -1, None, {'kind': 'source', 'unit': unit}, int(count))
            for unit, count in zip(session.unit_ids, session.manifest['counts'], strict=True)
        }
        return cls(
            nodes,
            copy.deepcopy(session.base_metadata),
            session.base_next_cluster_id,
            session.manifest.get('metadata_policy'),
            session.metadata_schema,
            object(),
        )

    @classmethod
    def from_session(cls, session):
        state = cls.initial(session)
        prefix = []
        for index, event in enumerate(session.events):
            state = state.advance(session, event, index, prefix)
            prefix.append(event)
        return state

    def restore_nodes(self, target, recursive):
        node = self.active.get(target)
        if node is None or node.parts is None:
            raise ValueError('No recoverable merge origin in this preparation for this live unit.')
        return node.index, leaves(node.parts) if recursive else node.parts

    def advance(self, session, event, index, prefix, allocator=None):
        """Validate one event against current state, without touching spike arrays."""
        from .live_summaries import _union

        operation = event.get('operation')
        if operation not in (
            'metadata_policy',
            'merge',
            'label',
            'threshold_split',
            'restore_parents',
        ):
            raise ValueError(f'Unsupported semantic operation: {operation!r}.')
        if operation == 'metadata_policy':
            # Migration is exceptional and must retain historical explicit-label intent.
            session._validate_event_capabilities([*prefix, event])
            _, metadata, minimum = interpret_events(
                session.unit_ids, session.base_metadata, [*prefix, event]
            )
            return SemanticState(
                self.active,
                metadata,
                max(minimum, self.next_id),
                POLICY,
                event['schema'],
                self.membership_key,
            )
        session._require_capability(operation, event.get('predicate'))
        expected_version = (3, 4) if operation == 'restore_parents' else (2 if self.policy else 1,)
        if type(event.get('version')) is not int or event['version'] not in expected_version:
            raise ValueError('Semantic event version disagrees with metadata policy.')
        children = [event.get('target')] if operation == 'merge' else event.get('children', [])
        if any(
            _integer(child, 'Allocated ID') < (self.next_id if allocator is None else allocator)
            for child in children
        ):
            raise ValueError('Allocated ID collides with the complete catalog allocator.')
        if operation == 'restore_parents':
            metadata, minimum = self._restore_metadata(session, event, prefix)
        else:
            # The public interpreter remains the full-history reference. For ordinary
            # operations its existing validator can also validate a single transition.
            _, metadata, minimum = interpret_events(list(self.active), self.metadata, [event])
        if operation == 'label':
            complete_schema(complete_schema(self.schema, self.metadata), metadata)
            return SemanticState(
                self.active,
                metadata,
                max(self.next_id, minimum),
                self.policy,
                self.schema,
                self.membership_key,
            )
        active = dict(self.active)
        if operation == 'merge':
            parts = tuple(active.pop(parent) for parent in event['parents'])
            counts = [p.count for p in parts]
            active[event['target']] = Node(
                event['target'],
                index,
                parts,
                _union([p.expression for p in parts]),
                sum(counts) if None not in counts else None,
            )
        elif operation == 'threshold_split':
            parent = active.pop(event['parent'])
            for selected, child in enumerate(event['children']):
                expression = {
                    'kind': 'split',
                    'parent': parent.expression,
                    'predicate': event['predicate'],
                    'selected': bool(selected),
                }
                active[child] = Node(child, index, None, expression, None)
        elif operation == 'restore_parents':
            _, parts = self.restore_nodes(event['target'], event['version'] == 4)
            active.pop(event['target'])
            active.update(dict(zip(event['children'], parts, strict=True)))
        return SemanticState(
            active, metadata, max(self.next_id, minimum), self.policy, self.schema, object()
        )

    def _restore_metadata(self, session, event, prefix):
        required = {
            'version',
            'operation',
            'target',
            'children',
            'merge_event_index',
            'merge_prefix_sha256',
            'source_revision',
            'metadata',
        } | ({'parent_references'} if event['version'] == 4 else set())
        if not self.policy or set(event) != required:
            raise ValueError('Malformed restore-parents event.')
        target = _integer(event['target'], 'Restore target ID')
        origin, parts = self.restore_nodes(target, event['version'] == 4)
        if _integer(event['merge_event_index'], 'Merge event index') != origin:
            raise ValueError('Restore requires an unchanged live recorded merge membership.')
        if event['merge_prefix_sha256'] != hash_json(prefix[: origin + 1]):
            raise ValueError('Restore merge prefix binding is invalid.')
        if event['source_revision'] != session.manifest['source']['revision']:
            raise ValueError('Restore source revision disagrees with preparation.')
        if event['version'] == 4:
            refs = event['parent_references']
            expected = [{'unit': p.unit, 'event_index': p.index} for p in parts]
            if (
                not isinstance(refs, list)
                or any(
                    not isinstance(ref, dict)
                    or set(ref) != {'unit', 'event_index'}
                    or type(ref['unit']) is not int
                    or type(ref['event_index']) is not int
                    for ref in refs
                )
                or refs != expected
            ):
                raise ValueError(
                    'Recursive restore parent references disagree with merge ancestry.'
                )
        children = [_integer(i, 'Restored child ID') for i in event['children']]
        if len(children) != len(parts) or len(set(children)) != len(children):
            raise ValueError('Restore requires one distinct child for every resolved parent.')
        if any(i > np.iinfo(np.int32).max for i in children):
            raise ValueError('Restored child ID exceeds the current Phy loader namespace.')
        resolved = normalize_metadata(event['metadata'])
        expected = {field: dict.fromkeys(children) for field in set(self.metadata) | {'group'}}
        if resolved != expected:
            raise ValueError('Restored membership must clear producer metadata and quality.')
        metadata = thaw(self.metadata)
        for values in metadata.values():
            values.pop(target, None)
        for field, values in resolved.items():
            metadata.setdefault(field, {}).update(values)
        return metadata, max(children) + 1


@dataclass(frozen=True)
class Transition:
    event: dict
    before: SemanticState
    after: SemanticState
    old_groups: dict
    new_groups: dict
    allocator: int

    def __post_init__(self):
        object.__setattr__(self, 'old_groups', freeze(self.old_groups))
        object.__setattr__(self, 'new_groups', freeze(self.new_groups))

    @property
    def nbytes(self):
        # Charge the full changed-node graph conservatively. Labels retain the
        # same membership map, so their accounting never traverses lineage.
        changed = set(self.old_groups) | set(self.new_groups)
        nodes = [
            state.active[unit]
            for state in (self.before, self.after)
            for unit in changed
            if unit in state.active
        ]
        return (
            4096
            + 256 * len(self.before.active)
            + retained_bytes(
                self.event,
                self.before.metadata,
                self.after.metadata,
                self.before.schema,
                self.after.schema,
                *nodes,
            )
            + sum(
                rows.nbytes
                for groups in (self.old_groups, self.new_groups)
                for rows in groups.values()
            )
        )
