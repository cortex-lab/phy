# -*- coding: utf-8 -*-
# flake8: noqa

"""Test CLI tool."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op

from click.testing import CliRunner

from .._misc import _write_text


#------------------------------------------------------------------------------
# Test CLI tool
#------------------------------------------------------------------------------

def test_cli_empty(temp_user_dir):
    # NOTE: make the import after the temp_user_dir fixture, to avoid
    # loading any user plugin affecting the CLI.
    from ..cli import phy

    runner = CliRunner()

    result = runner.invoke(phy, [])
    assert result.exit_code == 0

    result = runner.invoke(phy, ['--version'])
    assert result.exit_code == 0
    assert result.output.startswith('phy,')

    result = runner.invoke(phy, ['--help'])
    assert result.exit_code == 0
    assert result.output.startswith('Usage: phy')


def test_cli_plugins(temp_user_dir):

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
    path = op.join(temp_user_dir, 'plugins/hello.py')
    _write_text(path, cli_plugin)

    runner = CliRunner()

    # NOTE: make the import after the temp_user_dir fixture, to avoid
    # loading any user plugin affecting the CLI.
    from ..cli import phy

    result = runner.invoke(phy, ['--help'])
    assert result.exit_code == 0
    assert 'hello' in result.output
