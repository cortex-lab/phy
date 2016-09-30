# -*- coding: utf-8 -*-
# flake8: noqa

"""CLI tool."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import os
import os.path as op
import sys
from traceback import format_exception

import click
from six import exec_

from phy import (add_default_handler,
                 DEBUG, PDB, IPYTHON,
                 _Formatter, _logger_fmt,
                 __version_git__, discover_plugins)
from phy.utils import _fullname
from phy.utils.testing import _enable_pdb, _enable_profiler, _profile

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Set up logging with the CLI tool
#------------------------------------------------------------------------------

add_default_handler(level='DEBUG' if DEBUG else 'INFO')


def exceptionHandler(exception_type, exception, traceback):  # pragma: no cover
    logger.error("An error has occurred (%s): %s",
                 exception_type.__name__, exception)
    logger.debug(''.join(format_exception(exception_type,
                                          exception,
                                          traceback)))

# Only show traceback in debug mode (--debug).
# if not DEBUG:
sys.excepthook = exceptionHandler


def _add_log_file(filename):
    """Create a `phy.log` log file with DEBUG level in the
    current directory."""
    handler = logging.FileHandler(filename)

    handler.setLevel(logging.DEBUG)
    formatter = _Formatter(fmt=_logger_fmt,
                           datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)


def _run_cmd(cmd, ctx, glob, loc):  # pragma: no cover
    """Run a command with optionally a debugger, IPython, or profiling."""
    if PDB:
        _enable_pdb()
    if IPYTHON:
        from IPython import start_ipython
        args_ipy = ['-i', '--gui=qt']
        ns = glob.copy()
        ns.update(loc)
        return start_ipython(args_ipy, user_ns=ns)
    # Profiling. The builtin `profile` is added in __init__.
    prof = __builtins__.get('profile', None)
    if prof:
        prof = __builtins__['profile']
        return _profile(prof, cmd, glob, loc)
    return exec_(cmd, glob, loc)


#------------------------------------------------------------------------------
# CLI tool
#------------------------------------------------------------------------------

@click.group()
@click.version_option(version=__version_git__)
@click.help_option('-h', '--help')
@click.pass_context
def phy(ctx, pdb=None, ipython=None, prof=None, lprof=None):
    """By default, the `phy` command does nothing. Add subcommands with plugins
    using `attach_to_cli()` and the `click` library."""
    pass


#------------------------------------------------------------------------------
# CLI plugins
#------------------------------------------------------------------------------

def load_cli_plugins(cli, config_dir=None):
    """Load all plugins and attach them to a CLI object."""
    from .config import load_master_config

    config = load_master_config(config_dir=config_dir)
    plugins = discover_plugins(config.Plugins.dirs)

    for plugin in plugins:
        if not hasattr(plugin, 'attach_to_cli'):  # pragma: no cover
            continue
        logger.debug("Attach plugin `%s` to CLI.", _fullname(plugin))
        # NOTE: plugin is a class, so we need to instantiate it.
        try:
            plugin().attach_to_cli(cli)
        except Exception as e:  # pragma: no cover
            logger.error("Error when loading plugin `%s`: %s", plugin, e)


# Load all plugins when importing this module.
load_cli_plugins(phy)
