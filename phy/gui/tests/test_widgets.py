# -*- coding: utf-8 -*-

"""Test widgets."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from functools import partial
from pathlib import Path
from pytest import fixture, mark, raises

from phylib.utils import connect, unconnect
from phylib.utils.testing import captured_logging
import phy
from .test_qt import _block
from ..widgets import Table, IPythonView, KeyValueWidget


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

@fixture
def table(qtbot):
    columns = ["id", "count"]
    data = [{"id": i,
             "count": 100 - 10 * i,
             "float": float(i),
             "is_masked": True if i in (2, 3, 5) else False,
             } for i in range(10)]
    table = Table(
        columns=columns,
        data=data)
    table.show()
    assert table.columnCount() == 1

    yield table

    table.close()


#------------------------------------------------------------------------------
# Test key value widget
#------------------------------------------------------------------------------

def test_key_value_1(qtbot):
    widget = KeyValueWidget()
    widget.show()

    qtbot.addWidget(widget)
    qtbot.waitForWindowShown(widget)

    widget.add_pair("my text", "some text")
    widget.add_pair("my text multiline", "some\ntext", 'multiline')
    widget.add_pair("my float", 3.5)
    widget.add_pair("my int", 3)
    widget.add_pair("my bool", True)
    widget.add_pair("my list", [1, 5])

    widget.get_widget('my bool').setChecked(False)
    widget.get_widget('my list[0]').setValue(2)

    assert widget.to_dict() == {
        'my text': 'some text', 'my text multiline': 'some\ntext',
        'my float': 3.5, 'my int': 3, 'my bool': False, 'my list': [2, 5]}

    # qtbot.stop()
    widget.close()


#------------------------------------------------------------------------------
# Test IPython view
#------------------------------------------------------------------------------

@mark.filterwarnings("ignore")
def test_ipython_view_1(qtbot):
    view = IPythonView()
    view.show()
    view.start_kernel()
    view.stop()
    qtbot.wait(10)
    view.close()


@mark.filterwarnings("ignore")
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


#------------------------------------------------------------------------------
# Test table
#------------------------------------------------------------------------------

def test_table_empty_1(qtbot):
    table = Table()
    table.show()

    table.close()


def test_table_init_1(qtbot):
    data = [
        {'id': 0, 'a': 'b'},
        {'id': 1, 'a': 'c'},
        {'id': 3, 'u': 'd'},
    ]
    table = Table(data=data, columns=['id', 'a', 'u'])
    table.show()
    qtbot.addWidget(table)
    qtbot.waitForWindowShown(table)

    assert table._row2id(0) == 0
    assert table._row2id(1) == 1
    assert table._row2id(2) == 3
    with raises(ValueError):
        assert table._row2id(3) == -1

    assert table._get_value(0, 'a') == 'b'
    assert table._get_value(0, 'u') == ''
    assert table._get_value(1, 'a') == 'c'
    assert table._get_value(1, 'u') == ''
    with raises(ValueError):
        table._get_value(2, 'a')
    assert table._get_value(3, 'a') == ''
    assert table._get_value(3, 'u') == 'd'

    # qtbot.stop()
    table.close()


def test_table_init_2(qtbot):
    data = [
        {'id': 0, 'col0': '10'},
        {'id': 10, 'col0': '20'},
    ]
    table = Table(data=data, columns=['id', 'col0'])
    table.show()
    qtbot.addWidget(table)
    qtbot.waitForWindowShown(table)

    table.add([{'id': 20, 'col0': '30'}])

    assert table._get_value(0, 'col0') == '10'
    assert table._get_value(10, 'col0') == '20'
    assert table._get_value(20, 'col0') == '30'

    # qtbot.stop()
    table.close()


def test_table_invalid_column(qtbot):
    table = Table(data=[{'id': 0, 'a': 'b'}], columns=['id', 'u'])
    table.show()
    qtbot.addWidget(table)
    qtbot.waitForWindowShown(table)

    table.close()


def test_table_0(qtbot, table):
    assert len(table.get_selected()) == 0


def test_table_1(qtbot, table):
    table.select([1, 2])
    assert table.get_selected() == [1, 2]


def test_table_scroll(qtbot, table):
    table.add([{'id': 1000 + i, 'count': i} for i in range(1000)])
    table.scroll_to(1400)
    qtbot.stop()


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


def test_table_remove_all_and_add_2(qtbot, table):
    table.remove_all_and_add({"id": 1000})
    _assert(table.get_ids, [1000])


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


def test_table_filter(qtbot, table):
    table.filter("id == 5")
    _assert(table.get_ids, [5])

    table.filter("count == 80")
    _assert(table.get_ids, [2])

    table.filter()
    _assert(table.get_ids, list(range(10)))
