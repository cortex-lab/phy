# -*- coding: utf-8 -*-

"""Utility functions used for tests."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import sys
import time
from contextlib import contextmanager
from ..ext.six import StringIO


#------------------------------------------------------------------------------
# Utility functions
#------------------------------------------------------------------------------

@contextmanager
def captured_output():
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


#------------------------------------------------------------------------------
# Testing VisPy canvas
#------------------------------------------------------------------------------

def _frame(canvas):
    canvas.update()
    canvas.app.process_events()
    time.sleep(1. / 60.)


def show_test(canvas, n_frames=2):
    """Show a VisPy canvas for a fraction of second."""
    with canvas as c:
        show_test_run(c, n_frames)


def show_test_start(canvas):
    """This is the __enter__ of with canvas."""
    canvas.show()
    canvas._backend._vispy_warmup()


def show_test_run(canvas, n_frames=2):
    """Display frames of a canvas."""
    if n_frames == 0:
        while not canvas._closed:
            _frame(canvas)
    else:
        for _ in range(n_frames):
            _frame(canvas)
            if canvas._closed:
                return


def show_test_stop(canvas):
    """This is the __exit__ of with canvas."""
    # ensure all GL calls are complete
    if not canvas._closed:
        canvas._backend._vispy_set_current()
        canvas.context.finish()
        canvas.close()
    time.sleep(0.025)  # ensure window is really closed/destroyed


def show_colored_canvas(color, n_frames=5):
    """Show an emty VisPy canvas with a given background color for a fraction
    of second."""
    from vispy import app, gloo
    c = app.Canvas()

    @c.connect
    def on_paint(e):
        gloo.clear(color)

    show_test(c, n_frames=n_frames)
