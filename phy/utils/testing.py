# -*- coding: utf-8 -*-

"""Utility functions used for tests."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from contextlib import contextmanager
from cProfile import Profile
import functools
import logging
import os
import os.path as op
import sys
import time
from timeit import default_timer

from numpy.testing import assert_array_equal as ae
from numpy.testing import assert_allclose as ac
from six import StringIO
from six.moves import builtins

from ._types import _is_array_like
from .config import _ensure_dir_exists

logger = logging.getLogger(__name__)


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


@contextmanager
def captured_logging(name=None):
    buffer = StringIO()
    logger = logging.getLogger(name)
    handlers = logger.handlers
    for handler in logger.handlers:
        logger.removeHandler(handler)
    handler = logging.StreamHandler(buffer)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    yield buffer
    buffer.flush()
    logger.removeHandler(handler)
    for handler in handlers:
        logger.addHandler(handler)


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
        assert set(d_0) == set(d_1)
        for k_0 in d_0:
            _assert_equal(d_0[k_0], d_1[k_0])
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
    logger.info("%s took %.6fms.", name, duration / repeats)


class ContextualProfile(Profile):  # pragma: no cover
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


def _enable_profiler(line_by_line=False):  # pragma: no cover
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
    try:  # pragma: no cover
        from line_profiler import LineProfiler
        if isinstance(prof, LineProfiler):
            prof.print_stats()
        else:
            prof.print_stats('cumulative')
    except ImportError:  # pragma: no cover
        prof.print_stats('cumulative')
    sys.stdout = old_stdout
    stats = output.getvalue()
    # Stop capture.
    if 'Line' in prof.__class__.__name__:  # pragma: no cover
        fn = 'lstats.txt'
    else:
        fn = 'stats.txt'
    stats_file = op.join(dir, fn)
    with open(stats_file, 'w') as f:
        f.write(stats)


def _enable_pdb():  # pragma: no cover
    from IPython.core import ultratb
    logger.debug("Enabling debugger.")
    from PyQt4.QtCore import pyqtRemoveInputHook
    pyqtRemoveInputHook()
    sys.excepthook = ultratb.FormattedTB(mode='Verbose',
                                         color_scheme='Linux',
                                         call_pdb=True,
                                         )


#------------------------------------------------------------------------------
# Testing VisPy canvas
#------------------------------------------------------------------------------

def show_test(canvas):
    """Show a VisPy canvas for a fraction of second."""
    with canvas:
        # Interactive mode for tests.
        if 'PYTEST_INTERACT' in os.environ:  # pragma: no cover
            while not canvas._closed:
                canvas.update()
                canvas.app.process_events()
                time.sleep(1. / 60)
        else:
            canvas.update()
            canvas.app.process_events()


def show_colored_canvas(color):
    """Show a transient VisPy canvas with a uniform background color."""
    from vispy import app, gloo
    c = app.Canvas()

    @c.connect
    def on_draw(e):
        gloo.clear(color)

    show_test(c)
