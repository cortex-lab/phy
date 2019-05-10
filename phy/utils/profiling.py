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
from timeit import default_timer

from six import StringIO
from six.moves import builtins

from .config import _ensure_dir_exists

logger = logging.getLogger(__name__)


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
    from PyQt5.QtCore import pyqtRemoveInputHook
    pyqtRemoveInputHook()
    sys.excepthook = ultratb.FormattedTB(mode='Verbose',
                                         color_scheme='Linux',
                                         call_pdb=True,
                                         )


def memory_usage():
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss
