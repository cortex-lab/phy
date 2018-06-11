# -*- coding: utf-8 -*-

"""Test widgets."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op
from pytest import yield_fixture

from phy.utils.testing import captured_logging
from ..qt import block
from ..widgets import HTMLWidget, Table


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

@yield_fixture
def table(qtbot):
    columns = ["id", "count"]
    data = [{"id": i,
             "count": 100 - 10 * i,
             "_meta": "mask" if i in (2, 3, 5) else None,
             } for i in range(10)]
    table = Table(columns, data)
    table.show()
    qtbot.addWidget(table)
    qtbot.waitForWindowShown(table)

    yield table

    table.close()


#------------------------------------------------------------------------------
# Test widgets
#------------------------------------------------------------------------------

def test_widget_empty(qtbot):
    widget = HTMLWidget()
    widget.build()
    widget.show()
    qtbot.addWidget(widget)
    qtbot.waitForWindowShown(widget)
    # qtbot.stop()
    widget.close()


def test_widget_html(qtbot):
    widget = HTMLWidget()
    widget.builder.add_style('html, body, p {background-color: purple;}')
    path = op.join(op.dirname(__file__), '../static/styles.min.css')
    widget.builder.add_style_src(path)
    widget.builder.add_header('<!-- comment -->')
    widget.builder.set_body('Hello world!')
    widget.build()
    widget.show()
    qtbot.addWidget(widget)
    qtbot.waitForWindowShown(widget)
    assert 'Hello world!' in widget.html
    # qtbot.stop()
    widget.close()


def test_widget_javascript_1(qtbot):
    widget = HTMLWidget()
    widget.builder.add_script('var number = 1;')
    widget.build()
    widget.show()
    qtbot.addWidget(widget)
    qtbot.waitForWindowShown(widget)

    _out = []

    def _callback(res):
        _out.append(res)

    widget.eval_js('number', _callback, sync=False)
    block(lambda: _out == [1])

    # Test logging from JS.
    with captured_logging() as buf:
        widget.eval_js('console.log("hello world!");')
    assert 'hello world!' in buf.getvalue().lower()

    # qtbot.stop()
    widget.close()


def _test_widget_javascript_2(qtbot):
    # TODO: onWidgetReady
    widget = HTMLWidget()
    widget.builder.add_script("var l = [1, 2];")
    widget.builder.add_script('''
        onWidgetReady(function() {
            window.emit("test", l);
        });
    ''')

    _out = []

    @widget.connect_
    def on_test(arg):
        _out.append(arg)

    widget.build()
    widget.show()
    qtbot.addWidget(widget)
    qtbot.waitForWindowShown(widget)

    widget.block_until_loaded()
    assert _out == [[1, 2]]

    widget.unconnect_(on_test)
    # qtbot.stop()
    widget.close()


#------------------------------------------------------------------------------
# Test table
#------------------------------------------------------------------------------

def test_table_empty_1(qtbot):
    table = Table()
    table.show()
    qtbot.addWidget(table)
    qtbot.waitForWindowShown(table)
    table.close()


def test_table_invalid_column(qtbot):
    table = Table(data=[{'id': 0, 'a': 'b'}], columns=['id', 'u'])
    table.show()
    qtbot.addWidget(table)
    qtbot.waitForWindowShown(table)
    table.close()


def test_table_1(qtbot, table):
    table.select([1, 2])
    assert table.selected == [1, 2]
    # qtbot.stop()


def test_table_duplicates(qtbot, table):
    table.select([1, 1])
    assert table.selected == [1]
    # qtbot.stop()


def test_table_nav_first(qtbot, table):
    table.next()
    assert table.selected == [0]
    assert table.get_next_id() == 1


def test_table_nav_last(qtbot, table):
    table.previous()
    assert table.selected == [0]
    assert table.get_previous_id() is None


def test_table_nav_0(qtbot, table):
    table.select([4])

    table.next()
    assert table.selected == [6]

    table.previous()
    assert table.selected == [4]


def test_table_nav_1(qtbot, table):
    _sel = []
    assert table.selected == []

    @table.connect_
    def on_some_event(items, **kwargs):
        _sel.append(items)

    table.eval_js('table.emit("some_event", 123);')

    assert _sel == [123]

    # qtbot.stop()


def test_table_sort(qtbot, table):
    table.select([1])
    table.next()
    table.next()
    assert table.selected == [6]

    # Sort by count decreasing, and check that 0 (count 100) comes before
    # 1 (count 90). This checks that sorting works with number).
    table.sort_by('count', 'asc')

    assert table.current_sort == ('count', 'asc')
    assert table.selected == [6]
    assert table.get_ids() == list(range(9, -1, -1))

    table.next()
    assert table.selected == [4]

    table.sort_by('count', 'desc')
    assert table.get_ids() == list(range(10))

    # qtbot.stop()


def test_table_add_change_remove(qtbot, table):
    assert table.get_ids() == list(range(10))

    table.add({'id': 100, 'count': 1000})
    assert table.get_ids() == list(range(10)) + [100]

    table.remove([0, 1])
    assert table.get_ids() == list(range(2, 10)) + [100]

    assert table.get(100)['count'] == 1000
    table.change([{'id': 100, 'count': 2000}])
    assert table.get(100)['count'] == 2000


def test_table_change_and_sort_1(qtbot, table):
    table.change([{'id': 5, 'count': 1000}])
    assert table.get_ids() == list(range(10))


def test_table_change_and_sort_2(qtbot, table):
    table.sort_by('count', 'asc')
    assert table.get_ids() == list(range(9, -1, -1))

    # Check that the table is automatically resorted after a change.
    table.change([{'id': 5, 'count': 1000}])
    assert table.get_ids() == [9, 8, 7, 6, 4, 3, 2, 1, 0, 5]


def test_table_filter(qtbot, table):
    table.filter("id == 5")
    assert table.get_ids() == [5]

    table.filter("count == 80")
    assert table.get_ids() == [2]

    table.filter()
    assert table.get_ids() == list(range(10))
