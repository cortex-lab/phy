# -*- coding: utf-8 -*-

"""Trace GUI."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
from pathlib import Path

import numpy as np

from phylib.io.model import load_raw_data
from phylib.utils import Bunch

from phy.apps.template import get_template_params
from phy.cluster.views.trace import TraceView, select_traces
from phy.gui import create_app, run_app, GUI

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Trace GUI
#------------------------------------------------------------------------------

def create_trace_gui(dat_path, **kwargs):
    """Create the Trace GUI.

    Parameters
    ----------

    dat_path : str or Path
        Path to the raw data file.
    sample_rate : float
        The data sampling rate, in Hz.
    n_channels_dat : int
        The number of columns in the raw data file.
    dtype : str
        The NumPy data type of the raw binary file.

    """

    gui_name = 'TraceGUI'

    dat_path = Path(dat_path)

    # Support passing a params.py file.
    if dat_path.suffix == '.py':
        params = get_template_params(str(dat_path))
        return create_trace_gui(next(iter(params.pop('dat_path'))), **params)

    if dat_path.suffix == '.cbin':  # pragma: no cover
        data = load_raw_data(path=dat_path)
        sample_rate = data.sample_rate
        n_channels_dat = data.shape[1]
    else:
        sample_rate = float(kwargs['sample_rate'])
        assert sample_rate > 0.

        n_channels_dat = int(kwargs['n_channels_dat'])

        dtype = np.dtype(kwargs['dtype'])
        offset = int(kwargs['offset'] or 0)
        order = kwargs.get('order', None)

        # Memmap the raw data file.
        data = load_raw_data(
            path=dat_path,
            n_channels_dat=n_channels_dat,
            dtype=dtype,
            offset=offset,
            order=order,
        )

    duration = data.shape[0] / sample_rate

    create_app()
    gui = GUI(name=gui_name, subtitle=dat_path.resolve(), enable_threading=False)

    gui.set_default_actions()

    def _get_traces(interval):
        return Bunch(
            data=select_traces(
                data, interval, sample_rate=sample_rate))

    # TODO: load channel information

    view = TraceView(
        traces=_get_traces,
        n_channels=n_channels_dat,
        sample_rate=sample_rate,
        duration=duration,
        enable_threading=False,
    )
    view.attach(gui)

    return gui


def trace_gui(dat_path, **kwargs):  # pragma: no cover
    """Launch the Trace GUI.

    Parameters
    ----------

    dat_path : str or Path
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

    gui = create_trace_gui(dat_path, **kwargs)
    gui.show()
    run_app()
    gui.close()
