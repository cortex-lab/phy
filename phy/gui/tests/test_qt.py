# -*- coding: utf-8 -*-

"""Test Qt utilities."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from ..qt import (QtCore, QtGui, QtWebKit,
                  _set_qt_widget_position_size,
                  _button_name_from_enum,
                  _button_enum_from_name,
                  _prompt,
                  )


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

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


def test_prompt(qtbot):

    assert _button_name_from_enum(QtGui.QMessageBox.Save) == 'save'
    assert _button_enum_from_name('save') == QtGui.QMessageBox.Save

    box = _prompt("How are you doing?",
                  buttons=['save', 'cancel', 'close'],
                  )
    qtbot.mouseClick(box.buttons()[0], QtCore.Qt.LeftButton)
    assert 'save' in box.clickedButton().text().lower()
