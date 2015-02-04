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


def show_test(canvas, n_frames=5):
    """Show a VisPy canvas for a fraction of second."""
    with canvas as c:
        for _ in range(n_frames):
            c.update()
            c.app.process_events()
            time.sleep(1./60.)


def show_colored_canvas(color, n_frames=5):
    """Show an emty VisPy canvas with a given background color for a fraction
    of second."""
    from vispy import app, gloo
    c = app.Canvas()

    @c.connect
    def on_paint(e):
        gloo.clear(color)

    show_test(c, n_frames=n_frames)
