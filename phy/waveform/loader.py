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
        assert len(n_samples) == 2
        before, after = n_samples
        n_samples = before + after
    assert before >= 0
    assert after >= 0
    assert before + after == n_samples
    return before, after


def _slice(index, n_samples, margin=None):
    """Return a waveform slice."""
    if margin is None:
        margin = (0, 0)
    assert isinstance(n_samples, (tuple, list))
    assert len(n_samples) == 2
    before, after = n_samples
    assert isinstance(margin, (tuple, list))
    assert len(margin) == 2
    margin_before, margin_after = margin
    before += margin_before
    after += margin_after
    return slice(max(0, index - before), index + after, None)


class WaveformLoader(object):
    """Load waveforms from filtered or unfiltered traces."""

    def __init__(self, traces=None, offset=0, filter=None,
                 n_samples=None, filter_margin=0,
                 channels=None):
        # A (possibly memmapped) array-like structure with traces.
        if traces is not None:
            self.traces = traces
        else:
            self._traces = None
        # Offset of the traces: time (in samples) of the first trace sample.
        self._offset = 0
        # List of channels to use when loading the waveforms.
        self._channels = channels
        # A filter function that takes a (n_samples, n_channels) array as
        # input.
        self._filter = filter
        # Number of samples to return, can be an int or a
        # tuple (before, after).
        if n_samples is None:
            raise ValueError("'n_samples' must be specified.")
        self.n_samples_before_after = _before_after(n_samples)
        self.n_samples_waveforms = sum(self.n_samples_before_after)
        # Number of additional samples to use for filtering.
        self._filter_margin = _before_after(filter_margin)

    @property
    def traces(self):
        return self._traces

    @traces.setter
    def traces(self, value):
        self.n_samples_trace, self.n_channels_traces = value.shape
        self._traces = value

    @property
    def channels(self):
        return self._channels

    @channels.setter
    def channels(self, value):
        self._channels = value

    @property
    def n_channels_waveforms(self):
        if self._channels is not None:
            return len(self._channels)
        else:
            return self.n_channels_traces

    def load_at(self, time):
        """Load a waveform at a given time."""
        time_o = time - self._offset
        if not (0 <= time_o < self.n_samples_trace):
            raise ValueError("Invalid time {0:d}.".format(time_o))
        slice_extract = _slice(time_o,
                               self.n_samples_before_after,
                               self._filter_margin)
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

    def __getitem__(self, item):
        """Load a number of waveforms."""
        if isinstance(item, slice):
            raise NotImplementedError("Indexing with slices is not "
                                      "implemented yet.")
        # Ensure a list of time samples are being requested.
        spikes = item
        n_spikes = len(spikes)
        # Initialize the array.
        # TODO: int16
        shape = (n_spikes, self.n_samples_waveforms,
                 self.n_channels_waveforms)
        waveforms = np.empty(shape, dtype=np.float32)
        # Load all spikes.
        for i, time in enumerate(spikes):
            waveforms[i:i+1, ...] = self.load_at(time)
        # Make a subselection with the specified channels.
        if self._channels is not None:
            return waveforms[..., self._channels]
        else:
            return waveforms
