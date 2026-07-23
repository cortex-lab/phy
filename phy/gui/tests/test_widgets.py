"""Test widgets."""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

import sys
from functools import partial

from phylib.utils import connect, unconnect
from pytest import fixture, mark

from ..qt import QHeaderView, Qt
from ..widgets import Barrier, IPythonView, KeyValueWidget, Table
from . import show_and_wait
from .test_qt import _block

# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------


def _assert(f, expected):
    _out = []
    f(lambda x: _out.append(x))
    _block(lambda: _out == [expected])


def _wait_until_table_ready(qtbot, table):
    b = Barrier()
    connect(b(1), event='ready', sender=table)

    qtbot.addWidget(table)
    show_and_wait(qtbot, table)
    b.wait()


@fixture
def table(qtbot):
    columns = ['id', 'count']
    data = [
        {
            'id': i,
            'count': 100 - 10 * i,
            'float': float(i),
            'is_masked': i in (2, 3, 5),
        }
        for i in range(10)
    ]
    table = Table(columns=columns, value_names=['id', 'count', {'data': ['is_masked']}], data=data)
    _wait_until_table_ready(qtbot, table)

    yield table

    table.close()


# ------------------------------------------------------------------------------
# Test key value widget
# ------------------------------------------------------------------------------


def test_key_value_1(qtbot):
    widget = KeyValueWidget()
    qtbot.addWidget(widget)
    show_and_wait(qtbot, widget)

    widget.add_pair('my text', 'some text')
    widget.add_pair('my text multiline', 'some\ntext', 'multiline')
    widget.add_pair('my float', 3.5)
    widget.add_pair('my int', 3)
    widget.add_pair('my bool', True)
    widget.add_pair('my list', [1, 5])

    widget.get_widget('my bool').setChecked(False)
    widget.get_widget('my list[0]').setValue(2)

    assert widget.to_dict() == {
        'my text': 'some text',
        'my text multiline': 'some\ntext',
        'my float': 3.5,
        'my int': 3,
        'my bool': False,
        'my list': [2, 5],
    }

    # qtbot.stop()
    widget.close()


# ------------------------------------------------------------------------------
# Test IPython view
# ------------------------------------------------------------------------------


@mark.filterwarnings('ignore')
def test_ipython_view_1(qtbot):
    view = IPythonView()
    view.show()
    view.start_kernel()
    kernel = view.kernel
    view.stop()
    assert not kernel.iopub_thread.thread.is_alive()
    qtbot.wait(10)
    view.close()


@mark.filterwarnings('ignore')
def test_ipython_view_2(qtbot, tempdir):
    from ..gui import GUI

    gui = GUI(config_dir=tempdir)
    gui.set_default_actions()

    view = IPythonView()
    view.show()

    view.attach(gui)  # start the kernel and inject the GUI

    gui.show()
    view.dock.close()
    qtbot.wait(10)
    gui.close()
    qtbot.wait(10)


# ------------------------------------------------------------------------------
# Test table
# ------------------------------------------------------------------------------


def test_barrier_1(qtbot, table):
    table.select([1])

    b = Barrier()
    table.get_selected(b(1))
    table.get_next_id(b(2))
    assert not b.have_all_finished()

    @b.after_all_finished
    def after():
        assert b.result(1)[0][0] == [1]
        assert b.result(2)[0][0] == 4

    b.wait()
    assert b.result(1) and b.result(2)


def test_table_empty_1(qtbot):
    table = Table()
    _wait_until_table_ready(qtbot, table)
    assert table.debouncer
    table.close()


def test_table_debounce_delay(qtbot):
    table = Table(debounce_delay=17)
    _wait_until_table_ready(qtbot, table)
    assert table.debouncer.delay == 17
    table.close()


def test_table_invalid_column(qtbot):
    table = Table(data=[{'id': 0, 'a': 'b'}], columns=['id', 'u'])
    qtbot.addWidget(table)
    show_and_wait(qtbot, table)
    table.close()


def test_table_0(qtbot, table):
    assert table.filter_edit.focusPolicy() == Qt.NoFocus
    table.filter_edit.clearFocus()
    qtbot.mouseClick(table.filter_edit, Qt.LeftButton)
    assert table.filter_edit.hasFocus()

    table.filter_edit.setText('id >= 2')
    qtbot.keyClick(table.filter_edit, Qt.Key_Return)
    assert not table.filter_edit.hasFocus()

    qtbot.mouseClick(table.filter_edit, Qt.LeftButton)
    table.filter_edit.setText('id >= 3')
    qtbot.keyClick(table.filter_edit, Qt.Key_Escape)
    assert not table.filter_edit.hasFocus()
    assert table.filter_edit.text() == ''
    _assert(table.get_selected, [])

    qtbot.mouseClick(table.filter_edit, Qt.LeftButton)
    assert table.filter_edit.hasFocus()
    index = table._proxy.index(0, 0)
    qtbot.mouseClick(
        table.table_view.viewport(),
        Qt.LeftButton,
        pos=table.table_view.visualRect(index).center(),
    )
    assert not table.filter_edit.hasFocus()


def test_table_1(qtbot, table):
    assert table.is_ready()

    table.select([1, 2])
    _assert(table.get_selected, [1, 2])


def test_table_batch_update_fits_once(table):
    fit_calls = []
    table._fit_columns = lambda: fit_calls.append(True)

    with table.batch_update():
        table.add({'id': 10, 'count': 10})
        table.change({'id': 10, 'count': 11})
        table.remove([10])

    assert fit_calls == [True]


def test_table_add_remove_and_sparse_change(table):
    table.select([1, 2])
    table.add_remove([{'id': 10, 'count': 10}], [1, 3])

    assert table._model.row_by_id(1) is None
    assert table._model.row_by_id(3) is None
    assert table._model.row_by_id(10)['count'] == 10
    assert table.get_selected_ids() == [2]

    fit_calls = []
    table._fit_columns = lambda: fit_calls.append(True)
    table.change({'id': 10, 'count': 11})
    table.change({'id': 999, 'count': 12})

    assert table._model.row_by_id(10)['count'] == 11
    assert fit_calls == [True]


def test_table_row_height_is_fitted_once(qtbot):
    table = Table()
    _wait_until_table_ready(qtbot, table)
    fit_calls = []
    table.table_view.resizeRowsToContents = lambda: fit_calls.append(True)

    table.add({'id': 1})
    table.add({'id': 2})
    table.remove_all_and_add([{'id': 3}])

    assert fit_calls == [True]
    assert table.table_view.verticalHeader().sectionResizeMode(0) == QHeaderView.Fixed
    table.close()


def test_table_row_control_right_click(qtbot, table):
    clicked = []

    @connect(sender=table)
    def on_row_right_click(sender, row_id):
        clicked.append(row_id)

    table.select([1, 2])
    index = table._proxy.index(4, 0)
    pos = table.table_view.visualRect(index).center()

    # A plain right-click is non-mutating and reserved for a future context menu.
    qtbot.mouseClick(table.table_view.viewport(), Qt.RightButton, pos=pos)
    qtbot.wait(10)
    assert clicked == []
    _assert(table.get_selected, [1, 2])

    control_modifier = Qt.MetaModifier if sys.platform == 'darwin' else Qt.ControlModifier
    qtbot.mouseClick(table.table_view.viewport(), Qt.RightButton, control_modifier, pos=pos)
    _block(lambda: clicked == [4])
    _assert(table.get_selected, [1, 2])
    unconnect(on_row_right_click)


def test_table_scroll(qtbot, table):
    table.add([{'id': 1000 + i, 'count': i} for i in range(1000)])
    qtbot.wait(50)
    table.scroll_to(1400)


def test_table_busy(qtbot, table):
    table.select([1, 2])
    table.set_busy(True)
    _l = []

    def callback(out):
        _l.append(out)

    table.eval_js('table.debouncer.isBusy', callback=callback)
    _block(lambda: _l == [True])
    table.set_busy(False)


def test_table_duplicates(qtbot, table):
    table.select([1, 1])
    _assert(table.get_selected, [1])


def test_table_nav_first_1(qtbot, table):
    table.next()
    _assert(table.get_selected, [0])
    _assert(table.get_next_id, 1)


def test_table_nav_first_2(qtbot, table):
    table.first()
    _assert(table.get_selected, [0])
    _assert(table.get_next_id, 1)


def test_table_nav_last(qtbot, table):
    table.previous()
    _assert(table.get_selected, [0])
    _assert(table.get_previous_id, None)

    table.first()
    qtbot.wait(100)

    table.last()
    qtbot.wait(100)


def test_table_nav_0(qtbot, table):
    table.select([4])

    table.next()
    _assert(table.get_selected, [6])

    table.previous()
    _assert(table.get_selected, [4])


def test_table_navigation_skip_masked_policy(qtbot, table):
    # Masked rows are skipped by default, including by the callback-compatible
    # navigable-ID API. The unfiltered ID API remains unchanged.
    _assert(table.get_ids, list(range(10)))
    _assert(table.get_navigable_ids, [0, 1, 4, 6, 7, 8, 9])

    table.select([1])
    table.next()
    _assert(table.get_selected, [4])
    table.previous()
    _assert(table.get_selected, [1])

    table.first()
    _assert(table.get_selected, [0])
    table.previous()
    _assert(table.get_selected, [0])

    table.last()
    _assert(table.get_selected, [9])
    table.next()
    _assert(table.get_selected, [9])


def test_table_navigation_include_masked_and_runtime_toggle(qtbot, table):
    table.skip_masked = False
    _assert(table.get_navigable_ids, list(range(10)))

    table.select([1])
    table.next()
    _assert(table.get_selected, [2])
    table.next()
    _assert(table.get_selected, [3])

    # Changing the policy takes effect immediately and a manually selected
    # masked row remains a valid anchor for sibling navigation.
    table.skip_masked = True
    table.next()
    _assert(table.get_selected, [4])

    table.skip_masked = False
    table.previous()
    _assert(table.get_selected, [3])


def test_table_navigation_include_masked_from_constructor(qtbot):
    table = Table(
        value_names=['id', {'data': ['is_masked']}],
        data=[{'id': 0}, {'id': 1, 'is_masked': True}, {'id': 2}],
        skip_masked=False,
    )
    _wait_until_table_ready(qtbot, table)

    _assert(table.get_navigable_ids, [0, 1, 2])
    table.select([0])
    table.next()
    _assert(table.get_selected, [1])

    table.close()


def test_table_navigation_respects_visible_sort_and_filter(qtbot, table):
    table.sort_by('count', 'asc')
    table.filter('id >= 2 && id <= 7')
    _assert(table.get_ids, [7, 6, 5, 4, 3, 2])
    _assert(table.get_navigable_ids, [7, 6, 4])

    table.first()
    _assert(table.get_selected, [7])
    table.next()
    _assert(table.get_selected, [6])
    table.next()
    _assert(table.get_selected, [4])
    table.next()
    _assert(table.get_selected, [4])

    table.skip_masked = False
    _assert(table.get_navigable_ids, [7, 6, 5, 4, 3, 2])
    table.next()
    _assert(table.get_selected, [3])
    table.last()
    _assert(table.get_selected, [2])


def test_table_nav_1(qtbot, table):
    _sel = []

    @connect(sender=table)
    def on_some_event(sender, items, **kwargs):
        _sel.append(items)

    table.eval_js('table.emit("some_event", 123);')

    _block(lambda: _sel == [123])

    unconnect(on_some_event)


def test_table_sort(qtbot, table):
    table.select([1])
    table.next()
    table.next()
    _assert(table.get_selected, [6])

    _l = []

    @connect(sender=table)
    def on_table_sort(sender, row_ids):
        _l.append(row_ids)

    # Sort by count decreasing, and check that 0 (count 100) comes before
    # 1 (count 90). This checks that sorting works with number).
    table.sort_by('count', 'asc')

    _assert(table.get_current_sort, ['count', 'asc'])
    _assert(table.get_selected, [6])
    _assert(table.get_ids, list(range(9, -1, -1)))

    table.next()
    _assert(table.get_selected, [4])

    table.sort_by('count', 'desc')
    _assert(table.get_ids, list(range(10)))

    assert _l == [list(range(9, -1, -1)), list(range(10))]


def test_table_remove_all(qtbot, table):
    table.remove_all()
    _assert(table.get_ids, [])


def test_table_remove_all_and_add_1(qtbot, table):
    table.remove_all_and_add([])
    _assert(table.get_ids, [])


def test_table_remove_all_and_add_keeps_column_width(qtbot, table):
    header = table.table_view.horizontalHeader()
    width = header.sectionSize(1)

    table.remove_all_and_add({'id': 1000, 'count': 'x' * 100})
    qtbot.wait(1)

    _assert(table.get_ids, [1000])
    assert header.sectionSize(1) == width
    assert header.sectionResizeMode(1) == QHeaderView.Interactive


def test_table_remove_all_and_add_without_fitting(qtbot, table, monkeypatch):
    table.sort_by('count', 'asc')
    table.select([1])
    fit_calls = []
    monkeypatch.setattr(table, '_fit_columns', lambda: fit_calls.append(True))

    table.remove_all_and_add(
        [
            {'id': 20, 'count': 'a' * 100},
            {'id': 21, 'count': 'c' * 100},
            {'id': 22, 'count': 'b' * 100},
        ],
        fit_columns=False,
    )
    qtbot.wait(1)

    _assert(table.get_ids, [20, 22, 21])
    _assert(table.get_selected, [])
    _assert(table.get_current_sort, ['count', 'asc'])
    assert fit_calls == []

    table.remove_all_and_add([], fit_columns=False)
    qtbot.wait(1)

    _assert(table.get_ids, [])
    _assert(table.get_current_sort, ['count', 'asc'])
    assert fit_calls == []


def test_table_add_change_remove(qtbot, table):
    _assert(table.get_ids, list(range(10)))

    table.add({'id': 100, 'count': 1000})
    _assert(table.get_ids, list(range(10)) + [100])

    table.remove([0, 1])
    _assert(table.get_ids, list(range(2, 10)) + [100])

    _assert(partial(table.get, 100), {'id': 100, 'count': 1000})
    table.change([{'id': 100, 'count': 2000}])
    _assert(partial(table.get, 100), {'id': 100, 'count': 2000})


def test_table_change_and_sort_1(qtbot, table):
    table.change([{'id': 5, 'count': 1000}])
    _assert(table.get_ids, list(range(10)))


def test_table_change_and_sort_2(qtbot, table):
    table.sort_by('count', 'asc')
    _assert(table.get_ids, list(range(9, -1, -1)))

    # Check that the table is automatically resorted after a change.
    table.change([{'id': 5, 'count': 1000}])
    _assert(table.get_ids, [9, 8, 7, 6, 4, 3, 2, 1, 0, 5])


def test_table_change_metadata_preserves_sort(qtbot):
    data = [
        {'id': 0, 'count': 30, 'group': 'noise'},
        {'id': 1, 'count': 10, 'group': 'noise'},
        {'id': 2, 'count': 20, 'group': 'noise'},
    ]
    table = Table(
        columns=['id', 'count'],
        value_names=['id', 'count', {'data': ['group']}],
        data=data,
    )
    _wait_until_table_ready(qtbot, table)

    table.sort_by('count', 'desc')
    _assert(table.get_ids, [0, 2, 1])
    _assert(table.get_current_sort, ['count', 'desc'])

    table.change([{'id': 1, 'group': 'good'}])
    _assert(table.get_ids, [0, 2, 1])
    _assert(table.get_current_sort, ['count', 'desc'])

    table.close()


def test_table_filter(qtbot, table):
    table.filter('id == 5')
    _assert(table.get_ids, [5])

    table.filter('count == 80')
    _assert(table.get_ids, [2])

    table.filter()
    _assert(table.get_ids, list(range(10)))


def test_table_filter_comparison_operators(qtbot, table):
    table.filter('count > 80')
    _assert(table.get_ids, [0, 1])

    table.filter('count >= 80')
    _assert(table.get_ids, [0, 1, 2])

    table.filter('count < 80')
    _assert(table.get_ids, list(range(3, 10)))

    table.filter('count <= 80')
    _assert(table.get_ids, list(range(2, 10)))

    table.filter('id != 5')
    _assert(table.get_ids, [i for i in range(10) if i != 5])


def test_table_filter_combined_expression(qtbot, table):
    table.filter('(count >= 50) && id != 3')
    _assert(table.get_ids, [0, 1, 2, 4, 5])


def test_table_filter_or_expression(qtbot, table):
    table.filter('(id == 1) || (id == 3)')
    _assert(table.get_ids, [1, 3])


def test_table_filter_operator_precedence(qtbot, table):
    table.filter('(id == 1 || id == 3) && count < 90')
    _assert(table.get_ids, [3])


def test_table_filter_invalid_expression_shows_all_rows(qtbot, table):
    table.filter('id ===')
    _assert(table.get_ids, list(range(10)))


def test_table_filter_missing_field_shows_all_rows(qtbot, table):
    table.filter('missing == 1')
    _assert(table.get_ids, list(range(10)))


def test_table_filter_string_and_null_values(qtbot):
    data = [
        {'id': 0, 'group': 'good', 'label': None},
        {'id': 1, 'group': 'mua', 'label': 'x'},
        {'id': 2, 'group': 'noise', 'label': None},
    ]
    table = Table(
        columns=['id', 'label'],
        value_names=['id', 'label', {'data': ['group']}],
        data=data,
    )
    _wait_until_table_ready(qtbot, table)

    table.filter("group == 'good'")
    _assert(table.get_ids, [0])

    table.filter("group != 'noise'")
    _assert(table.get_ids, [0, 1])

    table.filter("label == 'x'")
    _assert(table.get_ids, [1])

    table.filter('label == null')
    _assert(table.get_ids, [0, 2])

    table.filter('label != null')
    _assert(table.get_ids, [1])

    table.close()


def test_table_filter_event_emits_visible_ids(qtbot, table):
    emitted = []

    @connect(sender=table)
    def on_table_filter(sender, row_ids):
        emitted.append(row_ids)

    table.filter('count >= 80')
    _assert(table.get_ids, [0, 1, 2])
    _block(lambda: emitted == [[0, 1, 2]])

    unconnect(on_table_filter)
