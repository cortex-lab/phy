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


def test_trace_gui_consecutive_parts(qtbot, tmp_path):
    import numpy as np
    from numpy.testing import assert_array_equal

    expected = np.arange(1000, dtype='int16').reshape(500, 2)
    expected[:200].tofile(tmp_path / 'first.bin')
    expected[200:].tofile(tmp_path / 'second.bin')
    params = tmp_path / 'params.py'
    params.write_text(
        "dat_path = ['first.bin', 'second.bin']\nsample_rate = 1000.\n"
        "n_channels_dat = 2\ndtype = 'int16'\noffset = 0\n"
    )
    gui = create_trace_gui(str(params))
    try:
        assert_array_equal(gui._trace_reader[:], expected)
    finally:
        gui.close()


def test_trace_gui_invalid_source(tmp_path):
    from pytest import raises

    params = tmp_path / 'params.py'
    params.write_text("dat_path = []\nsample_rate = 1000.\nn_channels_dat = 2\ndtype = 'int16'\n")
    with raises(ValueError, match='does not specify'):
        create_trace_gui(params)
    unknown = tmp_path / 'unknown.xyz'
    unknown.touch()
    with raises(ValueError, match='Cannot open'):
        create_trace_gui(str(unknown))
