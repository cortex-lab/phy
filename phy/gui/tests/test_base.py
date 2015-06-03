# -*- coding: utf-8 -*-1

"""Tests of base classes."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import raises

from ..base import WidgetCreator, BaseGUI, BaseSession
from ...utils import EventEmitter


#------------------------------------------------------------------------------
# Base tests
#------------------------------------------------------------------------------

def test_widget_creator():

    class MyWidget(EventEmitter):
        """Mock widget."""
        def __init__(self, param=None):
            super(MyWidget, self).__init__()
            self.name = 'my_widget'
            self._shown = False
            self.param = param

        @property
        def shown(self):
            return self._shown

        def close(self, e=None):
            self.emit('close', e)
            self._shown = False

        def show(self):
            self._shown = True

    widget_classes = {'my_widget': MyWidget}

    wc = WidgetCreator(widget_classes=widget_classes)
    assert not wc.get()
    assert not wc.get('my_widget')

    with raises(ValueError):
        wc.add('my_widget_bis')

    for show in (False, True):
        w = wc.add('my_widget', show=show)
        assert len(wc.get()) == 1
        assert len(wc.get('my_widget')) == 1

        assert w.shown is show
        w.show()
        assert w.shown

        w.close()
        assert not wc.get()
        assert not wc.get('my_widget')
