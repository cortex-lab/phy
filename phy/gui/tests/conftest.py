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
def gui(tempdir, qapp):
    gui = GUI(position=(200, 100), size=(100, 100), config_dir=tempdir)
    yield gui
    gui.close()


@yield_fixture
def actions(gui):
    yield Actions(gui)


@yield_fixture
def snippets(gui):
    yield Snippets(gui)
