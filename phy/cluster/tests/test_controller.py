# -*- coding: utf-8 -*-

"""Test controller."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op
from textwrap import dedent

from .conftest import MockController


#------------------------------------------------------------------------------
# Test controller
#------------------------------------------------------------------------------

def test_controller_1(qtbot, tempdir):

    plugin = dedent('''
    from phy import IPlugin

    class MockControllerPlugin(IPlugin):
        def attach_to_controller(self, controller):
            controller.hello = 'world'

    c = get_config()
    c.MockController.plugins = ['MockControllerPlugin']

    ''')
    with open(op.join(tempdir, 'phy_config.py'), 'w') as f:
        f.write(plugin)

    controller = MockController(config_dir=tempdir)
    gui = controller.create_gui()
    gui.show()

    # Ensure that the plugin has been loaded.
    assert controller.hello == 'world'

    controller.manual_clustering.select([2, 3])
    assert controller.get_mean_features(2) is not None
    assert len(controller.spikes_per_cluster(2)) > 0

    # qtbot.stop()
    gui.close()
