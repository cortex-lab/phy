# -*- coding: utf-8 -*-

"""Test Qt utilities."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import mark

from ..qt import QtWebKit, wrap_qt, _set_qt_widget_position_size


# Skip these tests in "make test-quick".
pytestmark = mark.long


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

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
    # NOTE: the new view must be yielded at the next iteration.
    view.close()
    view = QtWebKit.QWebView()
    _set_qt_widget_position_size(view, size=(100, 100))
    view.show()
    yield

    view.setHtml("finished")
    view.show()
    yield
