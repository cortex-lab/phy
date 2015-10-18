# -*- coding: utf-8 -*-

"""Test VisPy."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from vispy.app import Canvas, use_app, run
from pytest import yield_fixture


#------------------------------------------------------------------------------
# Utilities and fixtures
#------------------------------------------------------------------------------

@yield_fixture
def canvas(qapp):
    use_app('pyqt4')
    c = Canvas(keys='interactive')
    yield c
    c.close()
