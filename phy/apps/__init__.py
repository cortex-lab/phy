# -*- coding: utf-8 -*-

"""CLI tool."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from contextlib import contextmanager
import logging
from pathlib import Path
import sys
from traceback import format_exception

import click

from phylib import add_default_handler, _Formatter   # noqa
from phylib import _logger_date_fmt, _logger_fmt   # noqa

from phy import __version_git__
from phy.gui.qt import QtDialogLogger
from phy.utils.profiling import _enable_profiler, _enable_pdb

from .base import (  # noqa
    BaseController, WaveformMixin, FeatureMixin, TemplateMixin, TraceMixin)


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


#------------------------------------------------------------------------------
# Root CLI tool
#------------------------------------------------------------------------------

@click.group()
@click.version_option(version=__version_git__)
@click.help_option('-h', '--help')
@click.pass_context
def phycli(ctx):
    """Interactive visualization and manual spike sorting of large-scale ephys data."""
    add_default_handler(level='DEBUG' if DEBUG else 'INFO', logger=logging.getLogger('phy'))
    add_default_handler(level='DEBUG' if DEBUG else 'INFO', logger=logging.getLogger('phylib'))
    add_default_handler(level='DEBUG' if DEBUG else 'INFO', logger=logging.getLogger('mtscomp'))


#------------------------------------------------------------------------------
# GUI command wrapper
#------------------------------------------------------------------------------


def _gui_command(f):
    """Command options for GUI commands."""
    f = click.option(
        '--clear-cache/--no-clear-cache', default=False,
        help="Clear the .phy cache in the data directory.")(f)
    f = click.option(
        '--clear-state/--no-clear-state', default=False,
        help="Clear the GUI state in `~/.phy/` and in `.phy`.")(f)
    return f


#------------------------------------------------------------------------------
# Raw data GUI
#------------------------------------------------------------------------------

@phycli.command('trace-gui')  # pragma: no cover
@click.argument('dat-path', type=click.Path(exists=True))
@click.option('-s', '--sample-rate', type=float)
@click.option('-d', '--dtype', type=str)
@click.option('-n', '--n-channels', type=int)
@click.option('-h', '--offset', type=int)
@click.option('-f', '--fortran', type=bool, is_flag=True)
@_gui_command
@click.pass_context
def cli_trace_gui(ctx, dat_path, **kwargs):
    """Launch the trace GUI on a raw data file."""
    from .trace.gui import trace_gui
    with capture_exceptions():
        kwargs['n_channels_dat'] = kwargs.pop('n_channels')
        kwargs['order'] = 'F' if kwargs.pop('fortran', None) else None
        trace_gui(dat_path, **kwargs)


#------------------------------------------------------------------------------
# Template GUI
#------------------------------------------------------------------------------

@phycli.command('template-gui')  # pragma: no cover
@click.argument('params-path', type=click.Path(exists=True))
@_gui_command
@click.pass_context
def cli_template_gui(ctx, params_path, **kwargs):
    """Launch the template GUI on a params.py file."""
    from .template.gui import template_gui
    prof = __builtins__.get('profile', None)
    with capture_exceptions():
        if prof:
            from phy.utils.profiling import _profile
            return _profile(prof, 'template_gui(params_path)', globals(), locals())
        template_gui(params_path, **kwargs)


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
@_gui_command
@click.pass_context
def cli_kwik_gui(ctx, path, channel_group=None, clustering=None, **kwargs):
    """Launch the Kwik GUI on a Kwik file."""
    from .kwik.gui import kwik_gui
    with capture_exceptions():
        assert path
        kwik_gui(path, channel_group=channel_group, clustering=clustering, **kwargs)


@phycli.command('kwik-describe')
@click.argument('path', type=click.Path(exists=True))
@click.option('--channel-group', type=int, help='channel group')
@click.option('--clustering', type=str, help='clustering')
@click.pass_context
def cli_kwik_describe(ctx, path, channel_group=0, clustering='main'):
    """Describe a Kwik file."""
    from .kwik.gui import kwik_describe
    assert path
    kwik_describe(path, channel_group=channel_group, clustering=clustering)


#------------------------------------------------------------------------------
# Conversion
#------------------------------------------------------------------------------

@phycli.command('alf-convert')
@click.argument('subdirs', nargs=-1, type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument('out_dir', type=click.Path())
@click.pass_context
def cli_alf_convert(ctx, subdirs, out_dir):
    """Convert an ephys dataset into ALF. If several directories are specified, it is assumed
    that each directory contains the data for one probe of the same recording."""
    from phylib.io.alf import EphysAlfCreator
    from phylib.io.merge import Merger
    from phylib.io.model import load_model

    out_dir = Path(out_dir)

    if len(subdirs) >= 2:
        # Merge in the `merged` subdirectory inside the output directory.
        m = Merger(subdirs, out_dir / '_tmp_merged')
        model = m.merge()
    else:
        model = load_model(Path(subdirs[0]) / 'params.py')

    c = EphysAlfCreator(model)
    c.convert(out_dir)


#------------------------------------------------------------------------------
# Waveform extraction
#------------------------------------------------------------------------------

@phycli.command('extract-waveforms')
@click.argument('params-path', type=click.Path(exists=True))
@click.argument('n_spikes_per_cluster', type=int, default=500)
@click.option('--nc', type=int, default=16)
@click.pass_context
def template_extract_waveforms(
        ctx, params_path, n_spikes_per_cluster, nc=None):  # pragma: no cover
    """Extract spike waveforms."""
    from phylib.io.model import load_model

    model = load_model(params_path)
    model.save_spikes_subset_waveforms(
        max_n_spikes_per_template=n_spikes_per_cluster, max_n_channels=nc)
    model.close()
