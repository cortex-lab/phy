# -*- coding: utf-8 -*-

"""Test cluster views."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import fixture

from phy.gui import GUI


#------------------------------------------------------------------------------
# Utilities and fixtures
#------------------------------------------------------------------------------

@fixture
def gui(tempdir, qtbot):
    gui = GUI(position=(200, 200), size=(800, 600), config_dir=tempdir)
    gui.set_default_actions()
    gui.show()
    qtbot.wait(1)
    #qtbot.addWidget(gui)
    #qtbot.waitForWindowShown(gui)
    yield gui
    qtbot.wait(1)
    gui.close()
    qtbot.wait(1)
