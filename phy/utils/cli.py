# -*- coding: utf-8 -*-
# flake8: noqa

"""CLI tool."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import click

import phy
from phy.plugins import get_all_plugins


#------------------------------------------------------------------------------
# CLI tool
#------------------------------------------------------------------------------

@click.command()
@click.version_option(version=phy.__version_git__)
@click.help_option()
def phy():
    return 0


#------------------------------------------------------------------------------
# CLI plugins
#------------------------------------------------------------------------------

def load_cli_plugins(cli):
    """Load all plugins and attach them to a CLI object."""
    plugins = get_all_plugins()
    # TODO: try/except to avoid crashing if a plugin is broken.
    for plugin in plugins:
        # NOTE: plugin is a class, so we need to instantiate it.
        plugin().attach_to_cli(cli)


# Load all plugins for the phy CLI.
load_cli_plugins(phy)
