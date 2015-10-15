# -*- coding: utf-8 -*-
# flake8: noqa

"""Test CLI tool."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from click.testing import CliRunner

from ..cli import phy


#------------------------------------------------------------------------------
# Test CLI tool
#------------------------------------------------------------------------------

def test_cli():
    runner = CliRunner()

    result = runner.invoke(phy, [])
    assert result.exit_code == 0

    result = runner.invoke(phy, ['--version'])
    assert result.exit_code == 0
    assert result.output.startswith('phy,')

    result = runner.invoke(phy, ['--help'])
    assert result.exit_code == 0
    assert result.output.startswith('Usage: phy')
