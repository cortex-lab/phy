# -*- coding: utf-8 -*-
# flake8: noqa

"""CLI tool."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import sys

import click

import phy
from phy import add_default_handler, DEBUG

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# CLI tool
#------------------------------------------------------------------------------

add_default_handler('DEBUG' if DEBUG else 'INFO')


# Only show traceback in debug mode (--debug).
def exceptionHandler(exception_type, exception, traceback):  # pragma: no cover
    logger.error("%s: %s", exception_type.__name__, exception)


if not DEBUG:
    sys.excepthook = exceptionHandler


@click.group()
@click.version_option(version=phy.__version_git__)
@click.help_option('-h', '--help')
@click.pass_context
def phy(ctx):
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

load_cli_plugins(phy)
