# -*- coding: utf-8 -*-

"""Test widgets."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from random import random
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
             "count": int(random() * 100),
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


def test_widget_javascript_2(qtbot):
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

    _sel = []

    @table.connect_
    def on_select(items, **kwargs):
        _sel.append(items)

    table.eval_js('table.emit("select", [1]);')

    print(table.selected)
    # assert table.selected == [1]

    # qtbot.stop()


def test_table_sort(qtbot, table):
    table.select([1])

    # Sort by count decreasing, and check that 0 (count 100) comes before
    # 1 (count 90). This checks that sorting works with number).
    # table.sort_by('count', 'desc')

    table.next()
    assert table.selected == [4]
    # print(table.current_sort)
    # assert table.current_sort == ('count', 'desc')

    # qtbot.stop()
