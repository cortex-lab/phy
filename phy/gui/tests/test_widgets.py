# -*- coding: utf-8 -*-

"""Test widgets."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import yield_fixture, raises

from ..widgets import HTMLWidget, Table


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

@yield_fixture
def table(qtbot):
    table = Table()
    table.show()
    # qtbot.waitForWindowShown(table)

    def count(id):
        return 10000.5 - 10 * id
    table.add_column(count, show=True)

    def skip(id):
        return id == 4
    table.add_column(skip)

    table.set_rows(range(10))

    yield table

    table.close()


#------------------------------------------------------------------------------
# Test widgets
#------------------------------------------------------------------------------

def test_widget_empty(qtbot):
    widget = HTMLWidget()
    widget.show()
    # qtbot.waitForWindowShown(widget)
    # qtbot.stop()


def test_widget_html(qtbot):
    widget = HTMLWidget()
    widget.add_styles('html, body, p {background-color: purple;}')
    widget.add_header('<!-- comment -->')
    widget.set_body('Hello world!')
    widget.show()
    # qtbot.waitForWindowShown(widget)
    assert 'Hello world!' in widget.html()


def test_widget_javascript_1(qtbot):
    widget = HTMLWidget()
    widget.eval_js('number = 1;')
    widget.show()
    # qtbot.waitForWindowShown(widget)

    assert widget.eval_js('number') == 1


def test_widget_javascript_2(qtbot):
    widget = HTMLWidget()
    widget.show()
    # qtbot.waitForWindowShown(widget)
    _out = []

    @widget.connect_
    def on_test(arg):
        _out.append(arg)

    widget.eval_js('emit("test", [1, 2]);')
    assert _out == [[1, 2]]

    widget.unconnect_(on_test)

    # qtbot.stop()


#------------------------------------------------------------------------------
# Test table
#------------------------------------------------------------------------------

def test_table_current_sort():
    assert Table().current_sort == (None, None)


def test_table_default_sort(qtbot):
    table = Table()
    table.show()
    # qtbot.waitForWindowShown(table)

    with raises(ValueError):
        table.add_column(lambda _: _)

    def count(id):
        return 10000.5 - 10 * id
    table.add_column(count)
    table.set_default_sort('count', 'asc')
    table.set_rows(range(10))

    assert table.default_sort == ('count', 'asc')
    assert table.get_next_id() == 9
    table.next()
    assert table.selected == [9]

    table.sort_by('id', 'desc')
    table.set_rows(range(11))
    table.next()
    assert table.selected == [10]

    table.close()


def test_table_duplicates(qtbot, table):
    assert table.default_sort == (None, None)

    table.select([1, 1])
    assert table.selected == [1]
    # qtbot.stop()


def test_table_nav_first(qtbot, table):
    table.next()
    assert table.selected == [0]


def test_table_nav_last(qtbot, table):
    table.previous()
    assert table.selected == [9]


def test_table_nav_edge_0(qtbot, table):
    # The first item is skipped.
    table.set_rows([4, 5])
    table.next()
    assert table.selected == [5]


def test_table_nav_edge_1(qtbot, table):
    # The last item is skipped.
    table.set_rows([3, 4])
    assert table.get_previous_id() == 3
    table.previous()
    assert table.selected == [3]


def test_table_nav_0(qtbot, table):
    table.select([4])

    table.next()
    assert table.selected == [5]

    table.previous()
    assert table.selected == [3]

    _sel = []

    @table.connect_
    def on_select(items):
        _sel.append(items)

    table.eval_js('table.select([1]);')
    assert _sel == [[1]]

    assert table.selected == [1]

    # qtbot.stop()


def test_table_sort(qtbot, table):
    table.select([1])

    # Sort by count decreasing, and check that 0 (count 100) comes before
    # 1 (count 90). This checks that sorting works with number (need to
    # import tablesort.number.js).
    table.sort_by('count', 'desc')

    table.previous()
    assert table.selected == [0]

    assert table.current_sort == ('count', 'desc')

    # qtbot.stop()
