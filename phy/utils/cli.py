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

import phy
from phy import add_default_handler, DEBUG

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Set up logging with the CLI tool
#------------------------------------------------------------------------------

add_default_handler(level='DEBUG' if DEBUG else 'INFO')


def exceptionHandler(exception_type, exception, traceback):  # pragma: no cover
    logger.error("An error has occurred (%s): %s",
                 exception_type.__name__, exception)
    logger.debug('\n'.join(format_exception(exception_type,
                                            exception,
                                            traceback)))

# Only show traceback in debug mode (--debug).
# if not DEBUG:
sys.excepthook = exceptionHandler


# Create a `phy.log` log file with DEBUG level in the current directory.
def _add_log_file(filename):
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    formatter = phy._Formatter(fmt=phy._logger_fmt,
                               datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

_add_log_file(op.join(os.getcwd(), 'phy.log'))


#------------------------------------------------------------------------------
# CLI tool
#------------------------------------------------------------------------------

@click.group()
@click.version_option(version=phy.__version_git__)
@click.help_option('-h', '--help')
@click.pass_context
def phy(ctx):
    """By default, the `phy` command does nothing. Add subcommands with plugins
    using `attach_to_cli()` and the `click` library."""
    pass


#------------------------------------------------------------------------------
# CLI plugins
#------------------------------------------------------------------------------

def load_cli_plugins(cli):
    """Load all plugins and attach them to a CLI object."""
    from .config import load_master_config
    from .plugin import get_all_plugins

    config = load_master_config()
    plugins = get_all_plugins(config)

    # TODO: try/except to avoid crashing if a plugin is broken.
    for plugin in plugins:
        if not hasattr(plugin, 'attach_to_cli'):  # pragma: no cover
            continue
        logger.info("Attach plugin `%s` to CLI.", plugin.__name__)
        # NOTE: plugin is a class, so we need to instantiate it.
        plugin().attach_to_cli(cli)


# Load all plugins when importing this module.
load_cli_plugins(phy)
