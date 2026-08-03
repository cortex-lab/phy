"""Compact presentation table for automatic merge propositions.

This view owns only presentation state.  Proposition identity is always the
catalog's stable string key; the integer ``id`` used by :class:`Table` is an
ephemeral rendering implementation detail.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from phylib.utils import emit

from phy.gui.qt import QAbstractItemView, QColor
from phy.gui.widgets import Table


class MergePropositionsView(Table):
    """A compact, persistent table of merge-proposition projections.

    Rows supplied to :meth:`set_propositions` require a non-empty string
    ``key`` and may provide ``unit_ids``, ``status``, ``reason``, and
    ``new_unit_id``.  Clicking a row emits ``activate_merge_proposition`` with
    that stable key.  It does not itself alter curation or merge-workspace
    state.
    """

    _columns = ('proposition',)
    _status_colors = {
        'active': '#5ca8ff',
        'accepted': '#86d16d',
        'accepted_modified': '#e6ad4c',
        'rejected': '#888888',
        'stale': '#e58b3c',
        'invalid': '#e05a5a',
    }
    # Kept as a programmatic compatibility surface while the visible action
    # buttons are deliberately removed.  New callers should use activation and
    # navigation methods above; the Supervisor still owns these mutations.
    _action_events = {
        'review': 'review_merge_proposition',
        'reject': 'reject_merge_proposition',
        'skip': 'skip_merge_proposition',
        'reset': 'reset_merge_proposition',
    }

    def __init__(self, *args, data=None, **kwargs):
        super().__init__(
            *args,
            title='MERGE PROPOSITIONS',
            columns=self._columns,
            value_names=self._columns,
            data=[],
            sort=None,
            debounce_events=(),
            skip_masked=False,
            **kwargs,
        )
        self._key_by_id = {}
        self._id_by_key = {}
        self._current_key = None
        self.table_view.setSelectionMode(QAbstractItemView.SingleSelection)
        self.set_propositions(data or ())

    @property
    def current_key(self):
        """The locally highlighted proposition key, if any."""
        return self._current_key

    @staticmethod
    def _as_ordered_ids(value):
        if value is None:
            return ()
        if isinstance(value, (str, bytes)):
            return (value,)
        return tuple(value)

    @staticmethod
    def _format_proposition(unit_ids, new_unit_id=None, display_id=None):
        """Return the compact, scan-friendly proposition label."""
        labels = tuple(map(str, unit_ids))
        if len(labels) <= 4:
            text = ', '.join(labels)
        else:
            text = f'{labels[0]}, {labels[1]}, …, {labels[-1]} ({len(labels)})'
        if new_unit_id is not None and new_unit_id != '':
            text = f'{text} ⇒ {new_unit_id}'
        if display_id:
            text = f'{display_id} · {text}'
        return text

    def _normalize_row(self, row, index):
        if not isinstance(row, Mapping):
            raise TypeError('Merge proposition rows must be mappings.')
        key = row.get('key')
        if not isinstance(key, str) or not key:
            raise ValueError('Every merge proposition row requires a non-empty string key.')
        unit_ids = self._as_ordered_ids(row.get('unit_ids'))
        display_id = str(row.get('display_id') or f'P{index + 1}')
        status = str(row.get('status', 'pending'))
        new_unit_id = row.get('new_unit_id')
        reference = row.get('reference', unit_ids[0] if unit_ids else None)
        invalid_or_stale = status in {'invalid', 'stale'}
        full_proposition = ', '.join(map(str, unit_ids))
        if new_unit_id is not None and new_unit_id != '':
            full_proposition = f'{full_proposition} ⇒ {new_unit_id}'
        tooltip = f'{display_id} · {full_proposition}\nStatus: {status}'
        if reference is not None:
            tooltip = f'{tooltip}\nReference: {reference}'
        tooltip = f'{tooltip}\nKey: {key}'
        if row.get('reason'):
            tooltip = f'{tooltip}\n{row["reason"]}'
        return {
            'id': index,
            'proposition': self._format_proposition(unit_ids, new_unit_id, display_id),
            # Retain full metadata in the model for filtering, status text, and
            # stable-key recovery, but do not expose it as a table column.
            'key': key,
            'display_id': display_id,
            'unit_ids': unit_ids,
            'status': status,
            'reference': reference,
            'reason': row.get('reason') or '',
            'new_unit_id': new_unit_id,
            '_proposition_tooltip': tooltip,
            '_can_review': bool(
                row.get('can_review', status == 'pending' and not invalid_or_stale)
            ),
            '_can_reject': bool(
                row.get('can_reject', status == 'pending' and not invalid_or_stale)
            ),
            '_can_skip': bool(row.get('can_skip', status == 'pending' and not invalid_or_stale)),
            '_can_reset': bool(
                row.get(
                    'can_reset',
                    status in {'accepted', 'accepted_modified', 'rejected'}
                    and not invalid_or_stale,
                )
            ),
        }

    def set_propositions(self, rows: Iterable[Mapping]):
        """Replace the projection while retaining a still-present highlight."""
        previous_key = self._current_key
        normalized = [self._normalize_row(row, index) for index, row in enumerate(rows)]
        keys = [row['key'] for row in normalized]
        if len(set(keys)) != len(keys):
            raise ValueError('Merge proposition row keys must be unique.')
        self._key_by_id = {row['id']: row['key'] for row in normalized}
        self._id_by_key = {key: row_id for row_id, key in self._key_by_id.items()}
        self.remove_all_and_add(normalized, fit_columns=not self._column_widths_fitted)
        self._current_key = previous_key if previous_key in self._id_by_key else None
        if self._current_key is not None:
            self.set_selected_ids([self._id_by_key[self._current_key]])
            self._set_dock_status(self._current_key)

    def visible_keys(self):
        """Return stable keys in the table's current visible sorted order."""
        return tuple(self._key_by_id[row_id] for row_id in self._visible_ids())

    def actionable_keys(self):
        """Return visible pending/reviewable keys in their current table order."""
        return tuple(
            self._key_by_id[row_id]
            for row_id in self._visible_ids()
            if self._model.row_by_id(row_id).get('_can_review')
        )

    def is_actionable_key(self, key):
        row_id = self._id_by_key.get(key)
        row = self._model.row_by_id(row_id) if row_id is not None else None
        return bool(row and row.get('_can_review'))

    def can_trigger(self, action):
        """Whether the legacy programmatic action is valid for the current row."""
        if action not in self._action_events:
            raise ValueError(f'Unknown merge proposition action: {action}.')
        row_id = self._id_by_key.get(self._current_key)
        row = self._model.row_by_id(row_id) if row_id is not None else None
        return bool(row and row.get(f'_can_{action}'))

    def trigger(self, action):
        """Emit a legacy programmatic mutation intent for the current stable key."""
        if not self.can_trigger(action):
            return False
        emit(self._action_events[action], self, self._current_key)
        if action == 'skip':
            self.select_next_actionable()
        return True

    def select_key(self, key):
        """Highlight ``key`` locally, without emitting an activation event."""
        row_id = self._id_by_key.get(key)
        if row_id is None:
            return False
        self._current_key = key
        self.set_selected_ids([row_id])
        self.scroll_to(row_id)
        self._set_dock_status(key)
        return True

    def _select_actionable(self, direction):
        keys = self.actionable_keys()
        if not keys:
            return None
        if self._current_key not in keys:
            key = keys[0] if direction == 'next' else keys[-1]
        else:
            index = keys.index(self._current_key) + (1 if direction == 'next' else -1)
            index %= len(keys)
            key = keys[index]
        return key if self.select_key(key) else None

    def select_next_actionable(self):
        """Select the next visible actionable entry, wrapping once."""
        return self._select_actionable('next')

    def select_previous_actionable(self):
        """Select the previous visible actionable entry, wrapping once."""
        return self._select_actionable('previous')

    def _on_row_clicked(self, index):
        if not index.isValid():
            return
        visible_ids = self._visible_ids()
        if not 0 <= index.row() < len(visible_ids):
            return
        key = self._key_by_id[visible_ids[index.row()]]
        self.select_key(key)
        emit('activate_merge_proposition', self, key)

    def _set_dock_status(self, key):
        """Expose hidden metadata through the dock status line when available."""
        row_id = self._id_by_key.get(key)
        row = self._model.row_by_id(row_id) if row_id is not None else None
        dock = getattr(self, 'dock', None)
        if row is None or dock is None:
            return
        detail = f'{row["proposition"]} · {row["status"]}'
        if row['reference'] is not None:
            detail = f'{detail} · reference {row["reference"]}'
        if row['reason']:
            detail = f'{detail} · {row["reason"]}'
        dock.set_status(detail)

    def _foreground_color(self, row, column):
        """Tint complete rows by lifecycle state, like cluster-group rows."""
        color = self._status_colors.get(row.get('status'))
        return QColor(color) if color is not None else super()._foreground_color(row, column)
