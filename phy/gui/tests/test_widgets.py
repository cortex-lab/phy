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

    widget.show()
    qtbot.waitForWindowShown(widget)
    # qtbot.stop()


def test_table(qtbot):
    table = Table()

    table.set_body("""

        <table id="the-table" class="sort">
            <thead>
                <tr><th>id</th><th>count</th></tr>
            </thead>
            <tbody>
                <tr><td>1</td><td>20</td></tr>
                <tr><td>2</td><td>10</td></tr>
                <tr><td>3</td><td>30</td></tr>
            </tbody>
        </table>

        <script>
            new Tablesort(document.getElementById('the-table'));
        </script>

    """)

    table.show()
    qtbot.waitForWindowShown(table)
    # qtbot.stop()
