# -*- coding: utf-8 -*-

"""Test VisPy."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

# import sys

from vispy.app import Canvas, use_app
from pytest import yield_fixture  # , mark

# from phy.gui.qt import QObject, pyqtSignal


#------------------------------------------------------------------------------
# Utilities and fixtures
#------------------------------------------------------------------------------

# class ExceptionHandler(QObject):
#     errorSignal = pyqtSignal()
#     silentSignal = pyqtSignal()

#     def handler(self, exctype, value, traceback):
#         self.errorSignal.emit()
#         sys._excepthook(exctype, value, traceback)


# exceptionHandler = ExceptionHandler()
# sys._excepthook = sys.excepthook
# sys.excepthook = exceptionHandler.handler


# @mark.qt_no_exception_capture
@yield_fixture
def canvas(qapp):
    use_app('pyqt4')
    c = Canvas(keys='interactive')
    yield c
    c.close()
