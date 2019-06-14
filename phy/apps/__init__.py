# -*- coding: utf-8 -*-

"""CLI tool."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from contextlib import contextmanager
import logging
import sys
from traceback import format_exception

import click

from phylib import add_default_handler, _Formatter
from phylib import _logger_date_fmt, _logger_fmt

from phy import __version_git__
from phy.gui.qt import QtDialogLogger
from phy.utils.profiling import _enable_profiler, _enable_pdb


logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# CLI utils
#------------------------------------------------------------------------------

DEBUG = False
if '--debug' in sys.argv:  # pragma: no cover
    DEBUG = True
    sys.argv.remove('--debug')


if '--pdb' in sys.argv:  # pragma: no cover
    sys.argv.remove('--pdb')
    _enable_pdb()


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
    tb = ''.join(format_exception(exception_type, exception, traceback))
    logger.error("An error has occurred (%s): %s\n%s", exception_type.__name__, exception, tb)


@contextmanager
def capture_exceptions():  # pragma: no cover
    """Log exceptions instead of crashing the GUI, and display an error dialog on errors."""
    logger.debug("Start capturing exceptions.")

    # Add a custom exception hook.
    excepthook = sys.excepthook
    sys.excepthook = exceptionHandler

    # Add a dialog exception handler.
    handler = QtDialogLogger()
    handler.setLevel(logging.ERROR)
    logging.getLogger('phy').addHandler(handler)

    yield

    # Reset the original exception hook.
    sys.excepthook = excepthook

    # Remove the dialog exception handler.
    logging.getLogger('phy').removeHandler(handler)

    logger.debug("Stop capturing exceptions.")


def _add_log_file(filename):  # pragma: no cover
    """Create a `phy.log` log file with DEBUG level in the
    current directory."""
    handler = logging.FileHandler(filename)

    handler.setLevel(logging.DEBUG)
    formatter = _Formatter(fmt=_logger_fmt, datefmt=_logger_date_fmt)
    handler.setFormatter(formatter)
    logging.getLogger('phy').addHandler(handler)


#------------------------------------------------------------------------------
# Root CLI tool
#------------------------------------------------------------------------------

@click.group()
@click.version_option(version=__version_git__)
@click.help_option('-h', '--help')
@click.pass_context
def phycli(ctx):
    """Ephys data tool."""
    add_default_handler(level='DEBUG' if DEBUG else 'INFO', logger=logging.getLogger('phy'))
    add_default_handler(level='DEBUG' if DEBUG else 'INFO', logger=logging.getLogger('phylib'))


#------------------------------------------------------------------------------
# Template GUI
#------------------------------------------------------------------------------

@phycli.command('template-gui')  # pragma: no cover
@click.argument('params-path', type=click.Path(exists=True))
@click.pass_context
def cli_template_gui(ctx, params_path):
    """Launch the template GUI on a params.py file."""
    from .template.gui import template_gui
    prof = __builtins__.get('profile', None)
    with capture_exceptions():
        if prof:
            from phy.utils.profiling import _profile
            return _profile(prof, 'template_gui(params_path)', globals(), locals())
        template_gui(params_path)


@phycli.command('template-describe')
@click.argument('params-path', type=click.Path(exists=True))
@click.pass_context
def cli_template_describe(ctx, params_path):
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
    with capture_exceptions():
        kwik_gui(path, channel_group=channel_group, clustering=clustering)


@phycli.command('kwik-describe')
@click.argument('path', type=click.Path(exists=True))
@click.option('--channel-group', type=int, help='channel group')
@click.option('--clustering', type=str, help='clustering')
@click.pass_context
def cli_kwik_describe(ctx, path, channel_group=0, clustering='main'):
    """Describe a Kwik file."""
    from .kwik.gui import kwik_describe
    kwik_describe(path, channel_group=channel_group, clustering=clustering)


#------------------------------------------------------------------------------
# Conversion
#------------------------------------------------------------------------------

@phycli.command('alfconvert')
@click.argument('params-path', type=click.Path(exists=True))
@click.argument('out_dir', type=click.Path())
@click.pass_context
def cli_alf_convert(ctx, params_path, out_dir):
    """Describe a template file."""
    from phylib.io.alf import EphysAlfCreator
    from phylib.io.model import TemplateModel
    from phylib.utils._misc import read_python

    model = TemplateModel(**read_python(params_path))
    c = EphysAlfCreator(model)
    c.convert(out_dir)
