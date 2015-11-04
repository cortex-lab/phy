# -*- coding: utf-8 -*-

"""Test widgets."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import yield_fixture

from ..widgets import HTMLWidget, Table


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

@yield_fixture
def table(qtbot):
    table = Table()

    table.show()
    qtbot.waitForWindowShown(table)

    items = [{'id': i, 'count': 100 - 10 * i} for i in range(10)]
    items[4]['skip'] = True

    table.set_data(cols=['id', 'count'],
                   items=items,
                   )

    yield table

    table.close()


#------------------------------------------------------------------------------
# Test actions
#------------------------------------------------------------------------------

def test_widget_empty(qtbot):
    widget = HTMLWidget()
    widget.show()
    qtbot.waitForWindowShown(widget)
    # qtbot.stop()


def test_widget_html(qtbot):
    widget = HTMLWidget()
    widget.add_styles('html, body, p {background-color: purple;}')
    widget.add_header('<!-- comment -->')
    widget.set_body('Hello world!')
    widget.eval_js('widget.set_body("Hello from Javascript!");')
    widget.show()
    qtbot.waitForWindowShown(widget)
    widget.build()
    assert 'Javascript' in widget.html()


def test_widget_javascript(qtbot):
    widget = HTMLWidget()
    widget.show()
    qtbot.waitForWindowShown(widget)
    widget.eval_js('number = 1;')
    assert widget.get_js('number') == 1

    _out = []

    @widget.connect_
    def on_test(arg):
        _out.append(arg)

    widget.eval_js('emit("test", [1, 2]);')
    assert _out == [[1, 2]]

    widget.unconnect_(on_test)

    # qtbot.stop()


def test_table_nav(qtbot, table):
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
    table.sort_by('count')
    table.sort_by('count')

    table.previous()
    assert table.selected == [0]

    assert table.current_sort == ('count', 'desc')

    # qtbot.stop()
