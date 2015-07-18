# -*- coding: utf-8 -*-

"""Test Qt utilities."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import mark

from ..qt import (QtWebKit, QtGui,
                  qt_app,
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


def test_wrap(qtbot):

    view = QtWebKit.QWebView()

    def _assert(text):
        html = view.page().mainFrame().toHtml()
        assert html == '<html><head></head><body>' + text + '</body></html>'

    _set_qt_widget_position_size(view, size=(100, 100))
    view.setHtml("hello")
    qtbot.addWidget(view)
    qtbot.waitForWindowShown(view)
    view.show()
    _assert('hello')

    view.setHtml("world")
    _assert('world')
    view.close()

    view = QtWebKit.QWebView()
    _set_qt_widget_position_size(view, size=(100, 100))
    view.show()
    qtbot.addWidget(view)

    view.setHtml("finished")
    _assert('finished')
    view.close()


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
