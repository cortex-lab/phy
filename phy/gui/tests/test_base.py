# -*- coding: utf-8 -*-1

"""Tests of ba0se classes."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from ..base import WidgetCreator, BaseGUI, BaseSession
from ...utils import EventEmitter


#------------------------------------------------------------------------------
# Base tests
#------------------------------------------------------------------------------

def test_widget_creator():

    class MyWidget(EventEmitter):
        """Mock widget."""
        def __init__(self):
            self.name = 'My widget'
            self._shown = False

        def close(self):
            self.emit('close')
            self._shown = False

        def show(self):
            self._shown = True

    widget_classes = {'my_widget': MyWidget}

    wc = WidgetCreator(widget_classes=widget_classes)
