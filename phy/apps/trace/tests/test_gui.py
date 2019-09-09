# -*- coding: utf-8 -*-

"""Testing the Trace GUI."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging

from phylib.io.tests.conftest import template_path  # noqa

from ..gui import create_trace_gui

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_trace_gui_1(qtbot, template_path):  # noqa
    gui = create_trace_gui(template_path)
    gui.show()
    qtbot.addWidget(gui)
    qtbot.waitForWindowShown(gui)
    gui.close()
