# -*- coding: utf-8 -*-

"""Test gui."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import yield_fixture

from ..actions import Actions, Snippets
from ..gui import GUI


#------------------------------------------------------------------------------
# Utilities and fixtures
#------------------------------------------------------------------------------

@yield_fixture
def gui():
    yield GUI(position=(200, 100), size=(100, 100))


@yield_fixture
def actions():
    yield Actions()


@yield_fixture
def snippets():
    yield Snippets()
