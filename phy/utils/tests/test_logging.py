"""Unit tests for logging module."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import os

from ..logging import (StringLogger, ConsoleLogger, debug, info, warn,
                       FileLogger, register, unregister,
                       set_level)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_string_logger():
    l = StringLogger(fmt='')
    l.info("test 1")
    l.info("test 2")

    log = str(l)
    logs = log.split('\n')

    assert "test 1" in logs[0]
    assert "test 2" in logs[1]


def test_console_logger():
    l = ConsoleLogger(fmt='')
    l.info("test 1")
    l.info("test 2")

    l = ConsoleLogger(level='debug')
    l.info("test 1")
    l.info("test 2")


def test_file_logger():
    logfile = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'log.txt')
    l = FileLogger(logfile, fmt='', level='debug')
    l.debug("test file 1")
    l.debug("test file 2")
    l.info("test file info")
    l.warn("test file warn")
    l.close()

    with open(logfile, 'r') as f:
        contents = f.read()

    assert contents.strip().startswith("test file 1\ntest file 2")

    os.remove(logfile)


def test_register():
    l = StringLogger(fmt='')
    register(l)

    set_level('info')
    debug("test D1")
    info("test I1")
    warn("test W1")

    set_level('Debug')
    debug("test D2")
    info("test I2")
    warn("test W2")
    assert len(str(l).strip().split('\n')) == 5

    unregister(l)
