"""Test axes."""


# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

import os

from ..axes import Axes, format_time_ticks
from . import show_and_wait

# ------------------------------------------------------------------------------
# Tests axes
# ------------------------------------------------------------------------------


def test_axes_1(qtbot, canvas_pz):
    c = canvas_pz

    db = (0, -10, 1000, 10)
    g = Axes(data_bounds=db)
    g.attach(c)

    show_and_wait(qtbot, c)

    c.panzoom.zoom = 4
    c.panzoom.zoom = 8
    c.panzoom.pan = (3, 3)
    g.reset_data_bounds(db)

    g._update_zoom(c.panzoom.zoom)
    g._update_pan(c.panzoom.pan)

    if os.environ.get('PHY_TEST_STOP', None):  # pragma: no cover
        qtbot.stop()
    c.close()


def test_time_tick_formatting():
    assert format_time_ticks([0, 1000, 10000]) == ['0 s', '1,000 s', '10,000 s']
    assert format_time_ticks([0, 3600], unit='h') == ['0 h', '1 h']
    assert format_time_ticks([4000, 4800], unit='h', decimals=2) == ['1.11 h', '1.33 h']


def test_axes_x_formatter_survives_reset(qtbot, canvas_pz):
    c = canvas_pz
    axes = Axes(
        data_bounds=(0, 0, 3600, 1), format_x=lambda values: format_time_ticks(values, 'h')
    )
    axes.attach(c)
    axes.reset_data_bounds((0, 0, 7200, 1))
    assert all(label.endswith(' h') for label in axes.locator.xtext)
    c.close()
