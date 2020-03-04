# -*- coding: utf-8 -*-

"""Trace GUI."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging

from phylib.io.traces import get_ephys_reader
from phylib.utils import Bunch

from phy.apps.template import get_template_params
from phy.cluster.views.trace import TraceView, select_traces
from phy.gui import create_app, run_app, GUI

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Trace GUI
#------------------------------------------------------------------------------

def create_trace_gui(obj, **kwargs):
    """Create the Trace GUI.

    Parameters
    ----------

    obj : str or Path
        Path to the raw data file.
    sample_rate : float
        The data sampling rate, in Hz.
    n_channels_dat : int
        The number of columns in the raw data file.
    dtype : str
        The NumPy data type of the raw binary file.
    offset : int
        The header offset in bytes.

    """

    gui_name = 'TraceGUI'

    # Support passing a params.py file.
    if str(obj).endswith('.py'):
        params = get_template_params(str(obj))
        return create_trace_gui(next(iter(params.pop('dat_path'))), **params)

    kwargs = {
        k: v for k, v in kwargs.items()
        if k in ('sample_rate', 'n_channels_dat', 'dtype', 'offset')}
    traces = get_ephys_reader(obj, **kwargs)

    create_app()
    gui = GUI(name=gui_name, subtitle=obj.resolve(), enable_threading=False)
    gui.set_default_actions()

    def _get_traces(interval):
        return Bunch(
            data=select_traces(
                traces, interval, sample_rate=traces.sample_rate))

    # TODO: load channel information

    view = TraceView(
        traces=_get_traces,
        n_channels=traces.n_channels,
        sample_rate=traces.sample_rate,
        duration=traces.duration,
        enable_threading=False,
    )
    view.attach(gui)

    return gui


def trace_gui(obj, **kwargs):  # pragma: no cover
    """Launch the Trace GUI.

    Parameters
    ----------

    obj : str or Path
        Path to the raw data file
    sample_rate : float
        The data sampling rate, in Hz.
    n_channels_dat : int
        The number of columns in the raw data file.
    dtype : str
        The NumPy data type of the raw binary file.
    order : str
        Order of the data file: `C` or `F` (Fortran).

    """

    gui = create_trace_gui(obj, **kwargs)
    gui.show()
    run_app()
    gui.close()
