from __future__ import absolute_import
"""Logger utility classes and functions."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import sys
import logging
import traceback

from ..ext.six import iteritems, string_types


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def _get_log_format(debug=False):
    """Default logging format."""
    return '%(asctime)s  %(message)s'


def _get_caller():
    tb = traceback.extract_stack()[-6]
    module = os.path.splitext(os.path.basename(tb[0]))[0]
    line = str(tb[1])
    caller = "{0:s}:{1:s}".format(module, line)
    return caller.ljust(24)


# -----------------------------------------------------------------------------
# Stream classes
# -----------------------------------------------------------------------------

class StringStream(object):
    """Logger stream used to store all logs in a string."""
    def __init__(self):
        self.string = ""

    def write(self, line):
        self.string += line

    def flush(self):
        pass

    def __repr__(self):
        return self.string


# -----------------------------------------------------------------------------
# Logging classes
# -----------------------------------------------------------------------------

class Logger(object):
    """Save logging information to a stream."""
    def __init__(self, fmt=None, stream=None, level=None, name=None,
                 print_caller=True, handler=None):
        if stream is None:
            stream = sys.stdout
        if name is None:
            name = self.__class__.__name__
        self.name = name
        self.print_caller = print_caller
        if handler is None:
            self.stream = stream
            self.handler = logging.StreamHandler(self.stream)
        else:
            self.handler = handler
        self.level = level
        self.fmt = fmt
        # Set the level and corresponding formatter.
        self.set_level(level, fmt)

    def set_level(self, level=None, fmt=None):
        # Default level and format.
        if level is None:
            level = self.level or logging.INFO
        if isinstance(level, string_types):
            level = getattr(logging, level.upper())
        if fmt is None:
            fmt = self.fmt or _get_log_format(level == logging.DEBUG)
        # Create the Logger object.
        self._logger = logging.getLogger(self.name)
        # Create the formatter.
        formatter = logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S')
        self.handler.setFormatter(formatter)
        # Configure the logger.
        self._logger.setLevel(level)
        self._logger.propagate = False
        self._logger.addHandler(self.handler)

    def get_message(self, msg):
        msg = str(msg)
        if self.print_caller:
            return _get_caller() + msg
        else:
            return msg

    def close(self):
        pass

    def debug(self, msg):
        self._logger.debug(self.get_message(msg))

    def info(self, msg):
        self._logger.info(self.get_message(msg))

    def warn(self, msg):
        self._logger.warn(self.get_message(msg))


class StringLogger(Logger):
    """Log to a string."""
    def __init__(self, **kwargs):
        kwargs['stream'] = StringStream()
        super(StringLogger, self).__init__(**kwargs)

    def __repr__(self):
        return self.stream.__repr__()


class ConsoleLogger(Logger):
    """Log to the standard output."""
    def __init__(self, **kwargs):
        kwargs['stream'] = sys.stdout
        super(ConsoleLogger, self).__init__(**kwargs)


class FileLogger(Logger):
    """Log to a file."""
    def __init__(self, filename=None, **kwargs):
        kwargs['handler'] = logging.FileHandler(filename)
        super(FileLogger, self).__init__(**kwargs)

    def close(self):
        self.handler.close()
        self._logger.removeHandler(self.handler)
        del self.handler
        del self._logger


# -----------------------------------------------------------------------------
# Global variables
# -----------------------------------------------------------------------------

LOGGERS = {}


def register(logger):
    """Register a logger."""
    name = logger.name
    if name not in LOGGERS:
        LOGGERS[name] = logger


def unregister(logger):
    """Unregister a logger."""
    name = logger.name
    if name in LOGGERS:
        LOGGERS[name].close()
        del LOGGERS[name]


def _log(level, *msg):
    if isinstance(msg, tuple):
        msg = ' '.join(str(_) for _ in msg)
    for name, logger in iteritems(LOGGERS):
        getattr(logger, level)(msg)


def debug(*msg):
    """Generate a debug message."""
    _log('debug', *msg)


def info(*msg):
    """Generate an info message."""
    _log('info', *msg)


def warn(*msg):
    """Generate a warning."""
    _log('warn', *msg)


def set_level(level):
    """Set the level of all registered loggers.

    Parameters
    ----------

    level : str
        Can be `warn`, `info`, or `debug`.

    """
    for name, logger in iteritems(LOGGERS):
        logger.set_level(level)


def _default_logger(level='info'):
    """Create a default logger in `info` mode by default."""
    register(ConsoleLogger())
    set_level(level)
