"""Presentation-only table for automatic merge propositions.

The view intentionally accepts plain row dictionaries.  It has no dependency on
the proposition catalog or on a :class:`~phy.cluster.supervisor.Supervisor`, so
the GUI cannot become an alternative source of curation state.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from phylib.utils import emit

from phy.gui.qt import QAbstractItemView, QHBoxLayout, QPushButton
from phy.gui.widgets import Table


class MergePropositionsView(Table):
    """A persistent, local-selection table for merge-proposition projections.

    Rows supplied to :meth:`set_propositions` require a string ``key`` and may
    provide ``unit_ids``, ``status``, ``reason``, and ``new_unit_id``.  Button
    events carry only the stable proposition key:

    * ``review_merge_proposition``
    * ``reject_merge_proposition``
    * ``skip_merge_proposition``
    * ``reset_merge_proposition``

    Selecting a row is deliberately local presentation state: unlike a cluster
    table it never emits ``select`` and never starts a merge review.
    """

    _columns = (
        'key',
        'unit_ids',
        'status',
        'reason',
        'n_clusters',
        'reference',
        'new_unit_id',
    )
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
            columns=['id', *self._columns],
            value_names=['id', *self._columns],
            data=[],
            sort=('status', 'asc'),
            debounce_events=(),
            skip_masked=False,
            **kwargs,
        )
        self._key_by_id = {}
        self._id_by_key = {}
        self._current_key = None
        self.table_view.setSelectionMode(QAbstractItemView.SingleSelection)
        self._create_actions()
        self.set_propositions(data or ())

    @property
    def current_key(self):
        """The key of the locally highlighted proposition, if any."""
        return self._current_key

    def _create_actions(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.action_buttons = {}
        for action, label in (
            ('review', 'Review'),
            ('reject', 'Reject'),
            ('skip', 'Skip'),
            ('reset', 'Reset review'),
        ):
            button = QPushButton(label, self)
            button.setObjectName(f'merge-proposition-{action}')
            button.clicked.connect(lambda _checked=False, action=action: self.trigger(action))
            layout.addWidget(button)
            self.action_buttons[action] = button
        # The Table layout is [filter, table].  Put controls between them.
        self.layout().insertLayout(1, layout)
        self._update_action_buttons()

    @staticmethod
    def _as_ordered_ids(value):
        if value is None:
            return ()
        if isinstance(value, (str, bytes)):
            return (value,)
        return tuple(value)

    def _normalize_row(self, row, index):
        if not isinstance(row, Mapping):
            raise TypeError('Merge proposition rows must be mappings.')
        key = row.get('key')
        if not isinstance(key, str) or not key:
            raise ValueError('Every merge proposition row requires a non-empty string key.')
        unit_ids = self._as_ordered_ids(row.get('unit_ids'))
        status = str(row.get('status', 'pending'))
        invalid_or_stale = status in {'invalid', 'stale'}
        can_review = bool(row.get('can_review', not invalid_or_stale))
        can_reject = bool(row.get('can_reject', status == 'pending' and not invalid_or_stale))
        can_skip = bool(row.get('can_skip', status == 'pending' and not invalid_or_stale))
        can_reset = bool(
            row.get(
                'can_reset',
                status in {'accepted', 'accepted_modified', 'rejected'} and not invalid_or_stale,
            )
        )
        return {
            # Table itself uses integer IDs.  The catalog key remains a separate,
            # visible column and is recovered through ``_key_by_id`` after every
            # sort or filter operation.
            'id': index,
            'key': key,
            'unit_ids': unit_ids,
            'status': status,
            'reason': row.get('reason') or '',
            'n_clusters': len(unit_ids),
            'reference': unit_ids[0] if unit_ids else '',
            'new_unit_id': row.get('new_unit_id', ''),
            '_can_review': can_review,
            '_can_reject': can_reject,
            '_can_skip': can_skip,
            '_can_reset': can_reset,
        }

    def set_propositions(self, rows: Iterable[Mapping]):
        """Replace the table projection while retaining a surviving local row.

        The integer table row IDs are regenerated only for rendering.  Action
        identity is always recovered from the proposition key, so sorting and
        filtering cannot redirect a button action to another proposition.
        """
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
        self._update_action_buttons()

    def select_key(self, key):
        """Highlight ``key`` locally, without emitting a workflow event."""
        row_id = self._id_by_key.get(key)
        if row_id is None:
            return False
        self._current_key = key
        self.set_selected_ids([row_id])
        self.scroll_to(row_id)
        self._update_action_buttons()
        return True

    def _on_row_clicked(self, index):
        if not index.isValid():
            return
        row_id = self._visible_ids()[index.row()]
        self._current_key = self._key_by_id[row_id]
        self.set_selected_ids([row_id])
        self._update_action_buttons()

    def _row_for_current_key(self):
        row_id = self._id_by_key.get(self._current_key)
        return self._model.row_by_id(row_id) if row_id is not None else None

    def can_trigger(self, action):
        """Whether an explicit action is currently valid for the local row."""
        if action not in self._action_events:
            raise ValueError(f'Unknown merge proposition action: {action}.')
        row = self._row_for_current_key()
        return bool(row and row.get(f'_can_{action}'))

    def trigger(self, action):
        """Emit one explicit action for the currently highlighted proposition."""
        if not self.can_trigger(action):
            return False
        key = self._current_key
        emit(self._action_events[action], self, key)
        if action == 'skip':
            self._select_next_pending()
        return True

    def _select_next_pending(self):
        visible_ids = self._visible_ids()
        if not visible_ids:
            return
        current_id = self._id_by_key.get(self._current_key)
        start = visible_ids.index(current_id) + 1 if current_id in visible_ids else 0
        for row_id in visible_ids[start:] + visible_ids[:start]:
            row = self._model.row_by_id(row_id)
            if row and row.get('_can_skip'):
                self.select_key(self._key_by_id[row_id])
                return

    def _update_action_buttons(self):
        for action, button in getattr(self, 'action_buttons', {}).items():
            button.setEnabled(self.can_trigger(action))
