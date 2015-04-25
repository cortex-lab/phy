# -*- coding: utf-8 -*-

"""Plotting traces."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from vispy import gloo

from ._vispy_utils import (BaseSpikeVisual,
                           BaseSpikeCanvas,
                           )
from ..utils.array import _as_array
from ..utils.logging import debug


#------------------------------------------------------------------------------
# CCG visual
#------------------------------------------------------------------------------

class TraceVisual(BaseSpikeVisual):

    _shader_name = 'traces'
    _gl_draw_mode = 'line_strip'

    """TraceVisual visual."""
    def __init__(self, **kwargs):
        super(TraceVisual, self).__init__(**kwargs)
        self._traces = None

    # Data properties
    # -------------------------------------------------------------------------

    @property
    def traces(self):
        """Displayed traces."""
        return self._traces

    @traces.setter
    def traces(self, value):
        value = _as_array(value)
        assert value.ndim == 2
        self.n_samples, self.n_channels = value.shape
        self._traces = value
        self._empty = self.n_samples == 0
        self.set_to_bake('traces')

    # Data baking
    # -------------------------------------------------------------------------

    def _bake_traces(self):
        ns, nc = self.n_samples, self.n_channels

        a_index = np.c_[np.repeat(np.arange(nc), ns),
                        np.tile(np.arange(ns), nc)].astype(np.float32)

        self.program['a_position'] = self._traces.ravel().astype(np.float32)
        self.program['a_index'] = a_index
        self.program['n_channels'] = nc
        self.program['n_samples'] = ns
        self.program['u_scale'] = 1.0

        debug("bake traces", self._traces.shape)


class TraceView(BaseSpikeCanvas):
    _visual_class = TraceVisual

    def __init__(self, *args, **kwargs):
        super(TraceView, self).__init__(*args, **kwargs)
        self._pz.aspect = None
