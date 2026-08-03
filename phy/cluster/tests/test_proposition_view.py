"""Tests for the compact merge-proposition queue."""

from phylib.utils import connect, unconnect

from phy.gui.qt import Qt
from phy.gui.tests.test_widgets import _wait_until_table_ready

from .._proposition_view import MergePropositionsView


def test_merge_propositions_compact_projection_and_activation(qtbot):
    view = MergePropositionsView(
        data=[
            {
                'key': 'merge:1',
                'unit_ids': (1, 2, 3, 4, 5, 6),
                'status': 'accepted_modified',
                'reason': 'reviewed with an extra unit',
                'new_unit_id': 42,
            },
            {'key': 'merge:2', 'unit_ids': (7, 8), 'status': 'pending'},
        ]
    )
    _wait_until_table_ready(qtbot, view)

    assert view.columns == ['proposition']
    assert not hasattr(view, 'action_buttons')
    assert view._model.row_by_id(0)['proposition'] == '1, 2, …, 6 (6) ⇒ 42'
    index = view._model.index(0, 0)
    assert view._model.data(index, Qt.ToolTipRole) == (
        '1, 2, 3, 4, 5, 6 ⇒ 42\n'
        'Status: accepted_modified\n'
        'Reference: 1\n'
        'Key: merge:1\n'
        'reviewed with an extra unit'
    )
    assert view._model.data(index, Qt.ForegroundRole).name() == '#e6ad4c'

    activated = []

    @connect(event='activate_merge_proposition', sender=view)
    def on_activate(sender, key):
        activated.append(key)

    view._on_row_clicked(view._proxy_index_for_id(1))
    assert activated == ['merge:2']
    assert view.current_key == 'merge:2'
    unconnect(on_activate)
    view.close()
