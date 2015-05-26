# -*- coding: utf-8 -*-

"""Test utils plotting."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import mark

from vispy import app

from ...utils.testing import (show_test_start,
                              show_test_run,
                              show_test_stop,
                              )
from .._vispy_utils import LassoVisual
from .._panzoom import PanZoom, PanZoomGrid


# Skip these tests in "make test-quick".
pytestmark = mark.long()


#------------------------------------------------------------------------------
# Tests VisPy
#------------------------------------------------------------------------------

_N_FRAMES = 2


class TestCanvas(app.Canvas):
    _pz = None

    def __init__(self, visual, grid=False, **kwargs):
        super(TestCanvas, self).__init__(keys='interactive', **kwargs)
        self.visual = visual
        self._grid = grid
        self._create_pan_zoom()

    def _create_pan_zoom(self):
        if self._grid:
            self._pz = PanZoomGrid()
            self._pz.n_rows = self.visual.n_rows
        else:
            self._pz = PanZoom()
        self._pz.add(self.visual.program)
        self._pz.attach(self)

    def on_draw(self, event):
        """Draw the main visual."""
        self.context.clear()
        self.visual.draw()

    def on_resize(self, event):
        """Resize the OpenGL context."""
        self.context.set_viewport(0, 0, event.size[0], event.size[1])


def _show_visual(visual, grid=False, stop=True):
    view = TestCanvas(visual, grid=grid)
    show_test_start(view)
    show_test_run(view, _N_FRAMES)
    if stop:
        show_test_stop(view)
    return view


def test_lasso():
    lasso = LassoVisual()
    lasso.n_rows = 4
    lasso.box = (1, 3)
    lasso.points = [[+.8, +.8],
                    [-.8, +.8],
                    [-.8, -.8],
                    ]
    view = _show_visual(lasso, grid=True, stop=False)
    view.visual.add([+.8, -.8])
    show_test_run(view, _N_FRAMES)
    show_test_stop(view)
