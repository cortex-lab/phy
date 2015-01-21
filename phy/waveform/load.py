# -*- coding: utf-8 -*-

"""Load waveforms from traces."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ..ext import six


#------------------------------------------------------------------------------
# Waveform loader from traces
#------------------------------------------------------------------------------


def _before_after(n_samples):
    """Get the number of samples before and after."""
    if not isinstance(n_samples, (tuple, list)):
        before = n_samples // 2
        after = n_samples - before
    else:
        before, after = n_samples
    assert before >= 0
    assert after >= 0
    assert before + after == n_samples
    return before, after


def _slice(index, n_samples, margin=None):
    """Return a waveform slice."""
    if margin is None:
        margin = (0, 0)
    before, after = n_samples
    margin_before, margin_after = margin
    before += margin_before
    after += margin_after
    return slice(index - before, index + after, None)


class WaveformLoader(object):
    """Load waveforms from filtered or unfiltered traces."""

    def __init__(self, traces, spike_times, offset=0, filter=None,
                 n_samples=None, filter_margin=0):
        # A (possibly memmapped) array-like structure with traces.
        self._traces = traces
        self.n_samples_trace, self.n_channels = traces.shape
        # Offset of the traces: time (in samples) of the first trace sample.
        self._offset = 0
        # Array with all spike times.
        self._spike_times = spike_times
        # A filter function that takes a (n_samples, n_channels) array as
        # input.
        self._filter = filter
        # Number of samples to return, can be an int or a
        # tuple (before, after).
        if n_samples is None:
            raise ValueError("'n_samples' must be specified.")
        self._n_samples = _before_after(n_samples)
        # Number of additional samples to use for filtering.
        self._filter_margin = _before_after(filter_margin)

    @property
    def traces(self):
        return self._traces

    @traces.setter
    def traces(self, value):
        self._traces = value

    def load_at(self, time):
        """Load a waveform at a given time."""
        time_o = time - self._offset
        assert 0 <= time_o < self.n_samples_trace
        slice_extract = _slice(time_o, self._n_samples, self._filter_margin)
        extract = self._traces[slice_extract, :]
        # Filter the waveforms.
        if self._filter is not None:
            waveforms = self._filter(extract)
        else:
            waveforms = extract
        # Remove the margin.
        margin_before, margin_after = self._filter_margin
        if margin_after > 0:
            assert margin_before >= 0
            waveforms = waveforms[margin_before:-margin_after, :]
        return waveforms
