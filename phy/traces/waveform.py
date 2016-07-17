# -*- coding: utf-8 -*-

"""Waveform extraction."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging

import numpy as np
from scipy.interpolate import interp1d

from ..utils._types import _as_array, Bunch
from phy.io.array import _pad, _get_padded, _range_from_slice
from phy.traces.filter import apply_filter, bandpass_filter

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Waveform extractor from a connected component
#------------------------------------------------------------------------------

class WaveformExtractor(object):
    """Extract waveforms after data filtering and spike detection."""
    def __init__(self,
                 extract_before=None,
                 extract_after=None,
                 weight_power=None,
                 thresholds=None,
                 ):
        self._extract_before = extract_before
        self._extract_after = extract_after
        self._weight_power = weight_power if weight_power is not None else 1.
        self._thresholds = thresholds or {}

    def _component(self, component, data=None, n_samples=None):
        comp_s = component[:, 0]  # shape: (component_size,)
        comp_ch = component[:, 1]  # shape: (component_size,)

        # Get the temporal window around the waveform.
        s_min, s_max = (comp_s.min() - 3), (comp_s.max() + 4)
        s_min = max(s_min, 0)
        s_max = min(s_max, n_samples)
        assert s_min < s_max

        return Bunch(comp_s=comp_s,
                     comp_ch=comp_ch,
                     s_min=s_min,
                     s_max=s_max,
                     )

    def _normalize(self, x):
        x = _as_array(x)
        tw = self._thresholds['weak']
        ts = self._thresholds['strong']
        return np.clip((x - tw) / (ts - tw), 0, 1)

    def _comp_wave(self, data_t, comp):
        comp_s, comp_ch = comp.comp_s, comp.comp_ch
        s_min, s_max = comp.s_min, comp.s_max
        nc = data_t.shape[1]
        # Data on weak threshold crossings. shape: (some_length, nc)
        wave = np.zeros((s_max - s_min, nc), dtype=data_t.dtype)
        # The sample where the peak is reached, on each channel.
        wave[comp_s - s_min, comp_ch] = data_t[comp_s, comp_ch]
        return wave

    def masks(self, data_t, wave, comp):
        nc = data_t.shape[1]
        comp_ch = comp.comp_ch
        s_min = comp.s_min

        # Binary mask. shape: (nc,)
        masks_bin = np.zeros(nc, dtype=np.bool)
        masks_bin[np.unique(comp_ch)] = 1

        # Find the peaks (relative to the start of the chunk). shape: (nc,)
        peaks = np.argmax(wave, axis=0) + s_min
        # Peak values on each channel. shape: (nc,)
        peaks_values = data_t[peaks, np.arange(0, nc)] * masks_bin

        # Compute the float masks.
        masks_float = self._normalize(peaks_values)
        # Keep shank channels.
        return masks_float

    def spike_sample_aligned(self, wave, comp):
        s_min, s_max = comp.s_min, comp.s_max
        # Compute the fractional peak.
        wave_n = self._normalize(wave)
        wave_n_p = np.power(wave_n, self._weight_power)
        u = np.arange(s_max - s_min)[:, np.newaxis]
        # Spike aligned time relative to the start of the chunk.
        s_aligned = np.sum(wave_n_p * u) / np.sum(wave_n_p) + s_min
        return s_aligned

    def extract(self, data, s_aligned):
        s = int(s_aligned)
        # Get block of given size around peak sample.
        waveform = _get_padded(data,
                               s - self._extract_before - 1,
                               s + self._extract_after + 2)
        return waveform

    def align(self, waveform, s_aligned):
        s = int(s_aligned)
        sb, sa = self._extract_before, self._extract_after
        # Perform interpolation around the fractional peak.
        old_s = np.arange(s - sb - 1, s + sa + 2)
        new_s = np.arange(s - sb + 0, s + sa + 0) + (s_aligned - s)
        try:
            f = interp1d(old_s, waveform, bounds_error=True,
                         kind='cubic', axis=0)
        except ValueError:  # pragma: no cover
            logger.warn("Interpolation error at time %d", s)
            return waveform
        return f(new_s)

    def set_thresholds(self, **kwargs):
        self._thresholds.update(kwargs)

    def __call__(self, component=None, data=None, data_t=None):
        assert data.shape == data_t.shape
        comp = self._component(component,
                               data=data,
                               n_samples=data_t.shape[0],
                               )

        wave = self._comp_wave(data_t, comp)
        masks = self.masks(data_t, wave, comp)
        s_aligned = self.spike_sample_aligned(wave, comp)

        waveform_unaligned = self.extract(data, s_aligned)
        waveform_aligned = self.align(waveform_unaligned, s_aligned)

        assert waveform_aligned.ndim == 2
        assert masks.ndim == 1
        assert waveform_aligned.shape[1] == masks.shape[0]

        return s_aligned, masks, waveform_aligned


#------------------------------------------------------------------------------
# Waveform loader from traces (used in the manual sorting GUI)
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
    index = int(index)
    before = int(before)
    after = int(after)
    return slice(max(0, index - before), index + after, None)


class WaveformLoader(object):
    """Load waveforms from filtered or unfiltered traces."""

    def __init__(self,
                 traces=None,
                 sample_rate=None,
                 spike_samples=None,
                 masks=None,
                 mask_threshold=None,
                 filter_order=None,
                 n_samples_waveforms=None,
                 ):

        # Traces.
        if traces is not None:
            self.traces = traces
            self.n_samples_trace, self.n_channels = traces.shape
        else:
            self._traces = None
            self.n_samples_trace = self.n_channels = 0

        assert spike_samples is not None
        self._spike_samples = spike_samples
        self.n_spikes = len(spike_samples)

        self._masks = masks
        if masks is not None:
            assert self._masks.shape == (self.n_spikes, self.n_channels)
        self._mask_threshold = mask_threshold

        # Define filter.
        if filter_order:
            filter_margin = filter_order * 3
            b_filter = bandpass_filter(rate=sample_rate,
                                       low=500.,
                                       high=sample_rate * .475,
                                       order=filter_order,
                                       )
            self._filter = lambda x, axis=0: apply_filter(x, b_filter,
                                                          axis=axis)
        else:
            filter_margin = 0
            self._filter = lambda x, axis=0: x

        # Number of samples to return, can be an int or a
        # tuple (before, after).
        assert n_samples_waveforms is not None
        self.n_samples_before_after = _before_after(n_samples_waveforms)
        self.n_samples_waveforms = sum(self.n_samples_before_after)
        # Number of additional samples to use for filtering.
        self._filter_margin = _before_after(filter_margin)
        # Number of samples in the extracted raw data chunk.
        self._n_samples_extract = (self.n_samples_waveforms +
                                   sum(self._filter_margin))

        self.dtype = np.float32
        self.shape = (self.n_spikes, self._n_samples_extract, self.n_channels)
        self.ndim = 3

    @property
    def traces(self):
        """Raw traces."""
        return self._traces

    @traces.setter
    def traces(self, value):
        self.n_samples_trace, self.n_channels = value.shape
        self._traces = value

    @property
    def spike_samples(self):
        return self._spike_samples

    def _load_at(self, time, channels=None):
        """Load a waveform at a given time."""
        if channels is None:
            channels = slice(None, None, None)
        time = int(time)
        time_o = time
        ns = self.n_samples_trace
        if not (0 <= time_o < ns):
            raise ValueError("Invalid time {0:d}/{1:d}.".format(time_o, ns))
        slice_extract = _slice(time_o,
                               self.n_samples_before_after,
                               self._filter_margin)
        extract = self._traces[slice_extract][:, channels].astype(np.float32)

        # Pad the extracted chunk if needed.
        if slice_extract.start <= 0:
            extract = _pad(extract, self._n_samples_extract, 'left')
        elif slice_extract.stop >= ns - 1:
            extract = _pad(extract, self._n_samples_extract, 'right')

        assert extract.shape[0] == self._n_samples_extract
        return extract

    def __getitem__(self, spike_ids):
        """Load the waveforms of the specified spikes."""
        if isinstance(spike_ids, slice):
            spike_ids = _range_from_slice(spike_ids,
                                          start=0,
                                          stop=self.n_spikes,
                                          )
        if not hasattr(spike_ids, '__len__'):
            spike_ids = [spike_ids]

        # Ensure a list of time samples are being requested.
        spike_ids = _as_array(spike_ids)
        n_spikes = len(spike_ids)

        # Initialize the array.
        # NOTE: last dimension is time to simplify things.
        shape = (n_spikes, self.n_channels, self._n_samples_extract)
        waveforms = np.zeros(shape, dtype=np.float32)

        # No traces: return null arrays.
        if self.n_samples_trace == 0:
            return np.transpose(waveforms, (0, 2, 1))

        # Load all spikes.
        for i, spike_id in enumerate(spike_ids):
            assert 0 <= spike_id < self.n_spikes
            time = self._spike_samples[spike_id]

            # Find unmasked channels.
            if (self._masks is not None and
                    self._mask_threshold is not None):
                channels = self._masks[spike_id] >= self._mask_threshold
                channels = np.nonzero(channels)[0]
                if len(channels):
                    assert channels[-1] < self.n_channels
                nc = len(channels)
            else:
                channels = slice(None, None, None)
                nc = self.n_channels

            # Extract the waveforms on the unmasked channels.
            try:
                w = self._load_at(time, channels)
            except ValueError as e:  # pragma: no cover
                logger.warn("Error while loading waveform: %s", str(e))
                continue

            assert w.shape == (self._n_samples_extract, nc)

            waveforms[i, channels, :] = w.T

        # Filter the waveforms.
        waveforms_f = waveforms.reshape((-1, self._n_samples_extract))
        # Only filter the non-zero waveforms.
        unmasked = waveforms_f.max(axis=1) != 0
        waveforms_f[unmasked] = self._filter(waveforms_f[unmasked], axis=1)
        waveforms_f = waveforms_f.reshape((n_spikes, self.n_channels,
                                           self._n_samples_extract))

        # Remove the margin.
        margin_before, margin_after = self._filter_margin
        if margin_after > 0:
            assert margin_before >= 0
            waveforms_f = waveforms_f[:, :, margin_before:-margin_after]

        assert waveforms_f.shape == (n_spikes,
                                     self.n_channels,
                                     self.n_samples_waveforms,
                                     )

        # NOTE: we transpose before returning the array.
        return np.transpose(waveforms_f, (0, 2, 1))
