# -*- coding: utf-8 -*-

"""Test Qt utilities."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import raises

from ..qt import (QMessageBox, Qt, QWebView,
                  _button_name_from_enum,
                  _button_enum_from_name,
                  _prompt,
                  require_qt,
                  create_app,
                  QApplication,
                  )


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_require_qt_with_app():

    @require_qt
    def f():
        pass

    if not QApplication.instance():
        with raises(RuntimeError):  # pragma: no cover
            f()


def test_require_qt_without_app(qapp):

    @require_qt
    def f():
        pass

    # This should not raise an error.
    f()


def test_qt_app(qtbot):
    create_app()
    view = QWebView()
    qtbot.addWidget(view)
    view.close()


def test_web_view(qtbot):

    view = QWebView()

    def _assert(text):
        html = view.page().mainFrame().toHtml()
        assert html == '<html><head></head><body>' + text + '</body></html>'

    view.resize(100, 100)
    view.setHtml("hello")
    qtbot.addWidget(view)
    qtbot.waitForWindowShown(view)
    view.show()
    _assert('hello')

    view.setHtml("world")
    _assert('world')
    view.close()

    view = QWebView()
    view.resize(100, 100)
    view.show()
    qtbot.addWidget(view)

    view.setHtml("finished")
    _assert('finished')
    view.close()


def test_prompt(qtbot):

    assert _button_name_from_enum(QMessageBox.Save) == 'save'
    assert _button_enum_from_name('save') == QMessageBox.Save

    box = _prompt("How are you doing?",
                  buttons=['save', 'cancel', 'close'],
                  )
    qtbot.mouseClick(box.buttons()[0], Qt.LeftButton)
    assert 'save' in str(box.clickedButton().text()).lower()
