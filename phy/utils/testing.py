# -*- coding: utf-8 -*-

"""Utility functions used for tests."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import sys
import time
from contextlib import contextmanager
from timeit import default_timer
from cProfile import Profile
import os.path as op
import functools

from numpy.testing import assert_array_equal as ae
from numpy.testing import assert_allclose as ac

from ..ext.six import StringIO
from ..ext.six.moves import builtins
from ._types import _is_array_like
from .logging import info
from .settings import _ensure_dir_exists


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


def _assert_equal(d_0, d_1):
    """Check that two objects are equal."""
    # Compare arrays.
    if _is_array_like(d_0):
        try:
            ae(d_0, d_1)
        except AssertionError:
            ac(d_0, d_1)
    # Compare dicts recursively.
    elif isinstance(d_0, dict):
        assert sorted(d_0) == sorted(d_1)
        for (k_0, k_1) in zip(sorted(d_0), sorted(d_1)):
            assert k_0 == k_1
            _assert_equal(d_0[k_0], d_1[k_1])
    else:
        # General comparison.
        assert d_0 == d_1


#------------------------------------------------------------------------------
# Profiling
#------------------------------------------------------------------------------

@contextmanager
def benchmark(name='', repeats=1):
    start = default_timer()
    yield
    duration = (default_timer() - start) * 1000.
    info("{} took {:.6f}ms.".format(name, duration / repeats))


class ContextualProfile(Profile):
    def __init__(self, *args, **kwds):
        super(ContextualProfile, self).__init__(*args, **kwds)
        self.enable_count = 0

    def enable_by_count(self, subcalls=True, builtins=True):
        """ Enable the profiler if it hasn't been enabled before."""
        if self.enable_count == 0:
            self.enable(subcalls=subcalls, builtins=builtins)
        self.enable_count += 1

    def disable_by_count(self):
        """ Disable the profiler if the number of disable requests matches the
        number of enable requests.
        """
        if self.enable_count > 0:
            self.enable_count -= 1
            if self.enable_count == 0:
                self.disable()

    def __call__(self, func):
        """Decorate a function to start the profiler on function entry
        and stop it on function exit.
        """
        wrapper = self.wrap_function(func)
        return wrapper

    def wrap_function(self, func):
        """Wrap a function to profile it."""
        @functools.wraps(func)
        def wrapper(*args, **kwds):
            self.enable_by_count()
            try:
                result = func(*args, **kwds)
            finally:
                self.disable_by_count()
            return result
        return wrapper

    def __enter__(self):
        self.enable_by_count()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disable_by_count()


def _enable_profiler(line_by_line=False):
    if 'profile' in builtins.__dict__:
        return builtins.__dict__['profile']
    if line_by_line:
        import line_profiler
        prof = line_profiler.LineProfiler()
    else:
        prof = ContextualProfile()
    builtins.__dict__['profile'] = prof
    return prof


def _profile(prof, statement, glob, loc):
    dir = '.profile'
    dir = op.realpath(dir)
    _ensure_dir_exists(dir)
    prof.runctx(statement, glob, loc)
    # Capture stdout.
    old_stdout = sys.stdout
    sys.stdout = output = StringIO()
    try:
        from line_profiler import LineProfiler
        if isinstance(prof, LineProfiler):
            prof.print_stats()
        else:
            prof.print_stats('cumulative')
    except ImportError:
        prof.print_stats('cumulative')
    sys.stdout = old_stdout
    stats = output.getvalue()
    # Stop capture.
    stats_file = op.join(dir, 'stats.txt')
    with open(stats_file, 'w') as f:
        f.write(stats)


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
    """Show a transient VisPy canvas with a uniform background color."""
    from vispy import app, gloo
    c = app.Canvas()

    @c.connect
    def on_draw(e):
        gloo.clear(color)

    show_test(c, n_frames=n_frames)
