# -*- coding: utf-8 -*-

"""Test Qt utilities."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import mark

from ..qt import (QtWebKit, QtGui,
                  qt_app,
                  wrap_qt,
                  _set_qt_widget_position_size,
                  _prompt,
                  )
from ...utils.logging import set_level


# Skip these tests in "make test-quick".
pytestmark = mark.long


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def setup():
    set_level('debug')


def teardown():
    set_level('info')


@wrap_qt
def test_wrap():

    view = QtWebKit.QWebView()
    _set_qt_widget_position_size(view, size=(100, 100))
    view.setHtml("hello")
    view.show()
    yield

    view.setHtml("world")
    yield

    view.setHtml("!")
    yield

    # Close a view and open a new one.
    view.close()

    view = QtWebKit.QWebView()
    _set_qt_widget_position_size(view, size=(100, 100))
    view.show()
    yield

    view.setHtml("finished")
    yield

    view.close()
    yield


@mark.skipif()
def test_prompt():
    with qt_app():
        w = QtGui.QWidget()
        w.show()
        result = _prompt(w,
                         "How are you doing?",
                         buttons=['save', 'cancel', 'close'],
                         )
        print(result)
