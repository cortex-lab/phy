# -*- coding: utf-8 -*-

"""Test widgets."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from ..widgets import HTMLWidget


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
