"""Selection operations emitted by workflow tables.

This small value object deliberately has no Qt, GUI, or Supervisor
dependencies. A table can describe how its selected IDs changed without
making a controller infer an operation from the final IDs alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SelectionIntent(Enum):
    """The user operation which produced a table selection."""

    REPLACE = 'replace'
    EXTEND = 'extend'
    TOGGLE = 'toggle'
    CLEAR = 'clear'
    NAVIGATE = 'navigate'


@dataclass(frozen=True)
class SelectionMutation:
    """An immutable transition between two ordered row-ID selections."""

    intent: SelectionIntent
    before_ids: tuple[int, ...]
    after_ids: tuple[int, ...]
    added_ids: tuple[int, ...]
    removed_ids: tuple[int, ...]

    @classmethod
    def create(cls, intent, before_ids, after_ids):
        """Build a mutation and derive its ordered added/removed IDs."""
        before_ids = tuple(before_ids)
        after_ids = tuple(after_ids)
        before_set = set(before_ids)
        after_set = set(after_ids)
        return cls(
            intent=intent,
            before_ids=before_ids,
            after_ids=after_ids,
            added_ids=tuple(row_id for row_id in after_ids if row_id not in before_set),
            removed_ids=tuple(row_id for row_id in before_ids if row_id not in after_set),
        )

    def __post_init__(self):
        if not isinstance(self.intent, SelectionIntent):
            raise TypeError('intent must be a SelectionIntent.')
        for name in ('before_ids', 'after_ids', 'added_ids', 'removed_ids'):
            ids = tuple(getattr(self, name))
            if len(ids) != len(set(ids)):
                raise ValueError(f'{name} must contain unique cluster IDs.')
            object.__setattr__(self, name, ids)

        before_ids = self.before_ids
        after_ids = self.after_ids
        expected_added = tuple(row_id for row_id in after_ids if row_id not in before_ids)
        expected_removed = tuple(row_id for row_id in before_ids if row_id not in after_ids)
        if self.added_ids != expected_added or self.removed_ids != expected_removed:
            raise ValueError('Added and removed IDs must match the selection delta.')
