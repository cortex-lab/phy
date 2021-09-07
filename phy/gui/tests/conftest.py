# -*- coding: utf-8 -*-

"""Test gui."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import fixture

from ..actions import Actions, Snippets
from ..gui import GUI


#------------------------------------------------------------------------------
# Utilities and fixtures
#------------------------------------------------------------------------------

@fixture
def gui(tempdir, qtbot):
    gui = GUI(position=(200, 100), size=(100, 100), config_dir=tempdir)
    gui.set_default_actions()
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitForWindowShown(gui)
    yield gui
    gui.close()
    del gui


@fixture
def actions(gui):
    yield Actions(gui, name='actions')


@fixture
def snippets(gui):
    yield Snippets(gui)
