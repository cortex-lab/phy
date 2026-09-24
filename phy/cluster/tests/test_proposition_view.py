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
                'display_id': 'P12',
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
    assert view._model.row_by_id(0)['proposition'] == 'P12 · 1, 2, …, 6 (6) ⇒ 42'
    index = view._model.index(0, 0)
    assert view._model.data(index, Qt.ToolTipRole) == (
        'P12 · 1, 2, 3, 4, 5, 6 ⇒ 42\n'
        'Status: accepted_modified\n'
        'Reference: 1\n'
        'Key: merge:1\n'
        'reviewed with an extra unit'
    )
    assert view._model.data(index, Qt.ForegroundRole).name() == '#e6ad4c'
    assert view._foreground_color({'status': 'active'}, 'proposition') is None

    activated = []

    @connect(event='activate_merge_proposition', sender=view)
    def on_activate(sender, key):
        activated.append(key)

    view._on_row_clicked(view._proxy_index_for_id(1))
    assert activated == ['merge:2']
    assert view.current_key == 'merge:2'
    unconnect(on_activate)
    view.close()


def test_merge_propositions_active_row_changes_in_place(qtbot):
    view = MergePropositionsView(
        data=[
            {
                'key': f'merge:{index}',
                'unit_ids': (index, index + 1),
                'status': 'rejected' if index == 1 else 'pending',
                'catalog_status': 'rejected' if index == 1 else 'pending',
            }
            for index in range(20)
        ]
    )
    _wait_until_table_ready(qtbot, view)
    model = view._model
    resets = []
    model.modelReset.connect(lambda: resets.append(True))
    view.table_view.verticalScrollBar().setValue(5)
    scroll = view.table_view.verticalScrollBar().value()

    assert view.set_active_key('merge:10')
    assert view._model is model
    assert resets == []
    assert view._model.row_by_id(10)['status'] == 'active'
    assert view.table_view.verticalScrollBar().value() >= scroll

    assert view.set_active_key('merge:11', previous_key='merge:10')
    assert view._model is model
    assert view._model.row_by_id(10)['status'] == 'pending'
    assert view._model.row_by_id(11)['status'] == 'active'

    assert view.set_active_key('merge:1')
    assert view._model.row_by_id(11)['status'] == 'pending'
    assert view._model.row_by_id(1)['status'] == 'active'
    assert view._model.row_by_id(1)['_catalog_status'] == 'rejected'
    view.close()
