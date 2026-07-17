"""Testing the Trace GUI."""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

import logging

from phylib.io.tests.conftest import template_path  # noqa

from ..gui import create_trace_gui

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------


def test_trace_gui_1(qtbot, template_path):  # noqa
    gui = create_trace_gui(template_path)
    mmaps = list(gui._trace_reader._mmaps)
    qtbot.addWidget(gui)
    with qtbot.waitExposed(gui):
        gui.show()
    gui.close()
    assert all(arr._mmap.closed for arr in mmaps)
