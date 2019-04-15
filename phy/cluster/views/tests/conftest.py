# -*- coding: utf-8 -*-

"""Test cluster views."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import yield_fixture

from phy.gui import GUI


#------------------------------------------------------------------------------
# Utilities and fixtures
#------------------------------------------------------------------------------

@yield_fixture
def gui(tempdir, qtbot):
    gui = GUI(position=(200, 100), size=(100, 100), config_dir=tempdir)
    gui.show()
    qtbot.wait(1)
    #qtbot.addWidget(gui)
    #qtbot.waitForWindowShown(gui)
    yield gui
    qtbot.wait(1)
    gui.close()
    qtbot.wait(1)
