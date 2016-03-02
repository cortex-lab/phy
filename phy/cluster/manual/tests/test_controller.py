# -*- coding: utf-8 -*-

"""Test controller."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from .conftest import MockController


#------------------------------------------------------------------------------
# Test controller
#------------------------------------------------------------------------------

def test_controller_1(qtbot, tempdir):
    controller = MockController(tempdir)
    gui = controller.create_gui('MyGUI', config_dir=tempdir)
    gui.show()

    controller.manual_clustering.select([2, 3])

    # qtbot.stop()
    gui.close()
