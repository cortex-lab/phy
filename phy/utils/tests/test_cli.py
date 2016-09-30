# -*- coding: utf-8 -*-
# flake8: noqa

"""Test CLI tool."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op

from click.testing import CliRunner
import logging
from pytest import yield_fixture

from ..cli import _add_log_file
from .._misc import _write_text, _read_text

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Test CLI tool
#------------------------------------------------------------------------------

@yield_fixture
def runner():
    yield CliRunner()


def test_cli_empty(temp_config_dir, runner):

    # NOTE: make the import after the temp_config_dir fixture, to avoid
    # loading any user plugin affecting the CLI.
    from ..cli import phy, load_cli_plugins
    load_cli_plugins(phy)

    result = runner.invoke(phy, [])
    assert result.exit_code == 0

    result = runner.invoke(phy, ['--version'])
    assert result.exit_code == 0
    assert result.output.startswith('phy,')

    result = runner.invoke(phy, ['--help'])
    assert result.exit_code == 0
    assert result.output.startswith('Usage: phy')


def test_cli_plugins(temp_config_dir, runner):

    # Write a CLI plugin.
    cli_plugin = """
        import click
        from phy import IPlugin

        class MyPlugin(IPlugin):
            def attach_to_cli(self, cli):
                @cli.command()
                def hello():
                    click.echo("hello world")
    """
    path = op.join(temp_config_dir, 'plugins/hello.py')
    _write_text(path, cli_plugin)

    # NOTE: make the import after the temp_config_dir fixture, to avoid
    # loading any user plugin affecting the CLI.
    from ..cli import phy, load_cli_plugins
    load_cli_plugins(phy, config_dir=temp_config_dir)

    # The plugin should have added a new command.
    result = runner.invoke(phy, ['--help'])
    assert result.exit_code == 0
    assert 'hello' in result.output

    # The plugin should have added a new command.
    result = runner.invoke(phy, ['hello'])
    assert result.exit_code == 0
    assert result.output == 'hello world\n'


def test_add_log_file(tempdir):
    filename = op.join(tempdir, 'phy.log')
    _add_log_file(filename)
    logger.debug("test!")
    assert _read_text(filename).endswith("test!\n")
