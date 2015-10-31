# -*- coding: utf-8 -*-

"""Test widgets."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from ..widgets import HTMLWidget


#------------------------------------------------------------------------------
# Test actions
#------------------------------------------------------------------------------

def test_widget(qtbot):
    widget = HTMLWidget()
    widget.show()
    qtbot.waitForWindowShown(widget)
    # qtbot.stop()
