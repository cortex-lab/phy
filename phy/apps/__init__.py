# -*- coding: utf-8 -*-

"""CLI tool."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import os
import os.path as op
import shutil
import sys
from traceback import format_exception

import click

from phy import add_default_handler, __version_git__, _Formatter, _logger_fmt
from phy.utils.profiling import _enable_profiler


logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# CLI utils
#------------------------------------------------------------------------------

DEBUG = False
if '--debug' in sys.argv:  # pragma: no cover
    DEBUG = True
    sys.argv.remove('--debug')


PDB = False
if '--pdb' in sys.argv:  # pragma: no cover
    PDB = True
    sys.argv.remove('--pdb')


IPYTHON = False
if '--ipython' in sys.argv:  # pragma: no cover
    IPYTHON = True
    sys.argv.remove('--ipython')


# Add `profile` in the builtins.
if '--lprof' in sys.argv or '--prof' in sys.argv:  # pragma: no cover
    _enable_profiler('--lprof' in sys.argv)
    if '--prof' in sys.argv:
        sys.argv.remove('--prof')
    if '--lprof' in sys.argv:
        sys.argv.remove('--lprof')


#------------------------------------------------------------------------------
# Set up logging with the CLI tool
#------------------------------------------------------------------------------

def exceptionHandler(exception_type, exception, traceback):  # pragma: no cover
    logger.error("An error has occurred (%s): %s",
                 exception_type.__name__, exception)
    logger.debug(''.join(format_exception(exception_type,
                                          exception,
                                          traceback)))


# Only show traceback in debug mode (--debug).
# if not DEBUG:
#sys.excepthook = exceptionHandler


def _add_log_file(filename):
    """Create a `phy.log` log file with DEBUG level in the
    current directory."""
    handler = logging.FileHandler(filename)

    handler.setLevel(logging.DEBUG)
    formatter = _Formatter(fmt=_logger_fmt,
                           datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logging.getLogger('phy').addHandler(handler)


def _copy_gui_state(gui_name, module_name, config_dir=None):
    """Copy the state.json file."""
    config_dir = config_dir or op.join(op.realpath(op.expanduser('~')), '.phy')
    gui_dir = op.join(config_dir, gui_name)
    if not op.exists(gui_dir):
        os.makedirs(gui_dir)
    # Create the script if it doesn't already exist.
    path = op.join(gui_dir, 'state.json')
    if op.exists(path):
        return
    curdir = op.dirname(op.realpath(__file__))
    from_path = op.join(curdir, '../apps', module_name, 'static', 'state.json')
    logger.debug("Copy %s to %s" % (from_path, path))
    shutil.copy(from_path, path)


#------------------------------------------------------------------------------
# Root CLI tool
#------------------------------------------------------------------------------

@click.group()
@click.version_option(version=__version_git__)
@click.help_option('-h', '--help')
@click.pass_context
def phycli(ctx, pdb=None, ipython=None, prof=None, lprof=None):
    """By default, the `phy` command does nothing. Add subcommands with plugins
    using `attach_to_cli()` and the `click` library."""
    add_default_handler(level='DEBUG' if DEBUG else 'INFO')


#------------------------------------------------------------------------------
# Template GUI
#------------------------------------------------------------------------------

@phycli.command('template-gui')  # pragma: no cover
@click.argument('params-path', type=click.Path(exists=True))
def cli_template_gui(params_path):
    """Launch the template GUI on a params.py file."""
    from .template.gui import template_gui
    template_gui(params_path)


@phycli.command('template-describe')
@click.argument('params-path', type=click.Path(exists=True))
def cli_template_describe(params_path):
    """Describe a template file."""
    from .template.gui import template_describe
    template_describe(params_path)


#------------------------------------------------------------------------------
# Kwik GUI
#------------------------------------------------------------------------------

# Create the `phy cluster-manual file.kwik` command.
@phycli.command('kwik-gui')  # pragma: no cover
@click.argument('path', type=click.Path(exists=True))
@click.option('--channel-group', type=int)
@click.option('--clustering', type=str)
@click.pass_context
def cli_kwik_gui(ctx, path, channel_group=None, clustering=None):
    """Launch the Kwik GUI on a Kwik file."""
    from .kwik.gui import kwik_gui
    kwik_gui(path, channel_group=channel_group, clustering=clustering)


@phycli.command('kwik-describe')
@click.argument('path', type=click.Path(exists=True))
@click.option('--channel-group', type=int, help='channel group')
@click.option('--clustering', type=str, help='clustering')
def cli_kwik_describe(path, channel_group=0, clustering='main'):
    """Describe a Kwik file."""
    from .kwik.gui import kwik_describe
    kwik_describe(path, channel_group=channel_group, clustering=clustering)
