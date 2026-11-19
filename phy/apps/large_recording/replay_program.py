"""Batch projection compiled once from an already validated semantic journal."""

from copy import deepcopy

import numpy as np

from .curation import _membership_mask, _restore_batch_size, evaluate_predicate
from .live_summaries import membership_expressions
from .streaming import UnitRows


class AssignmentProgram:
    """Collapse merge runs around explicit split/restore boundaries.

    Compilation is independent of spike count. Every structural event remains in
    order, including predicates with rejection obligations whose children later
    remerge. Restoration always uses immutable original assignments/attributes.
    The reference journal interpreter remains independent of this implementation.
    """

    def __init__(self, unit_ids, events):
        events = deepcopy(events)
        active = set(map(int, unit_ids))
        self.steps = []
        self.predicate_names = {
            event['predicate']['attribute']
            for event in events
            if event['operation'] == 'threshold_split'
        }
        restores = {}
        if any(event['operation'] == 'restore_parents' for event in events):
            membership_expressions(unit_ids, events, restore_sources=restores)
        mapping = {unit: unit for unit in active}

        def flush():
            if any(old != target for old, target in mapping.items()):
                keys = np.array(sorted(mapping), dtype=np.int64)
                targets = np.array([mapping[int(key)] for key in keys], dtype=np.int64)
                self.steps.append(('map', UnitRows(keys), targets))

        for index, event in enumerate(events):
            operation = event['operation']
            if operation == 'merge':
                parents = set(event['parents'])
                target = event['target']
                mapping = {
                    old: target if value in parents else value for old, value in mapping.items()
                }
                active.difference_update(parents)
                active.add(target)
            elif operation in ('threshold_split', 'restore_parents'):
                flush()
                if operation == 'threshold_split':
                    self.steps.append(('split', event))
                    active.remove(event['parent'])
                else:
                    expressions = restores[index]
                    size = _restore_batch_size(expressions, len(self.predicate_names))
                    self.steps.append(('restore', event, expressions, size))
                    active.remove(event['target'])
                active.update(event['children'])
                mapping = {unit: unit for unit in active}
            elif operation not in ('label', 'metadata_policy'):
                raise ValueError('Unsupported semantic projection operation.')
        flush()

    @property
    def merge_runs(self):
        return sum(step[0] == 'map' for step in self.steps)

    def project(self, assignments, attributes=None):
        original = np.asarray(assignments).reshape(-1)
        output = original.astype(np.int64, copy=True)
        attributes = attributes or {}
        for step in self.steps:
            if step[0] == 'map':
                _, lookup, targets = step
                output = targets[lookup(output)]
            elif step[0] == 'split':
                event = step[1]
                predicate = event['predicate']
                name = predicate['attribute']
                if name not in attributes:
                    raise ValueError(f'Full-source threshold attribute is unavailable: {name}.')
                values = np.asarray(attributes[name]).reshape(-1)
                if len(values) != len(output):
                    raise ValueError('Threshold attribute is not a scalar per-spike array.')
                rows = np.flatnonzero(output == event['parent'])
                selected = evaluate_predicate(values[rows], predicate)
                output[rows] = np.where(selected, event['children'][1], event['children'][0])
            else:
                _, event, expressions, size = step
                rows = np.flatnonzero(output == event['target'])
                for start in range(0, len(rows), size):
                    chunk = rows[start : start + size]
                    base = original[chunk]
                    values = {
                        name: np.asarray(attributes[name]).reshape(-1)[chunk]
                        for name in self.predicate_names
                        if name in attributes
                    }
                    covered = np.zeros(len(chunk), dtype=bool)
                    memo = {}
                    for child, expression in zip(event['children'], expressions, strict=True):
                        mask = _membership_mask(expression, base, values, memo)
                        if np.any(covered & mask):
                            raise ValueError('Restored source parent memberships overlap.')
                        output[chunk[mask]] = child
                        covered |= mask
                    if not covered.all():
                        raise ValueError('Restored source parents do not partition the target.')
        return output
