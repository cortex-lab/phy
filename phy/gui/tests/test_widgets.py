# -*- coding: utf-8 -*-

"""Test widgets."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from ..widgets import HTMLWidget, Table


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
    # qtbot.stop()


def test_widget_javascript(qtbot):
    widget = HTMLWidget()
    widget.show()
    qtbot.waitForWindowShown(widget)
    widget.eval_js('number = 1;')
    assert widget.get_js('number') == 1
    # qtbot.stop()


def test_table(qtbot):
    table = Table()

    table.show()
    qtbot.waitForWindowShown(table)

    items = [{'id': i, 'count': 10 * i} for i in range(10)]
    items[4]['skip'] = True

    table.set_data(cols=['id', 'count'],
                   items=items,
                   )
    table.select([4])

    assert table.pinned == []

    table.next()
    assert table.selected == [5]

    table.previous()
    assert table.selected == [3]
