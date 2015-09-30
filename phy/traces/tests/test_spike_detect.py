# -*- coding: utf-8 -*-

"""Tests of spike detection routines."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_array_equal as ae
from pytest import yield_fixture

from phy.utils.datasets import download_test_data
from phy.utils.tests.test_context import context, ipy_client
from phy.electrode import load_probe
from ..spike_detect import (SpikeDetector,
                            _spikes_to_keep,
                            _trim_spikes,
                            _add_chunk_offset,
                            _concat_spikes,
                            )


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

@yield_fixture
def traces():
    path = download_test_data('test-32ch-10s.dat')
    traces = np.fromfile(path, dtype=np.int16).reshape((200000, 32))
    traces = traces[:20000]

    yield traces


@yield_fixture(params=[(True,), (False,)])
def spike_detector(request):
    remap = request.param[0]

    probe = load_probe('1x32_buzsaki')
    channel_mapping = {i: i for i in range(1, 21, 2)} if remap else None

    sd = SpikeDetector()
    sd.use_single_threshold = False
    sample_rate = 20000
    sd.set_metadata(probe,
                    channel_mapping=channel_mapping,
                    sample_rate=sample_rate)

    yield sd


#------------------------------------------------------------------------------
# Test spike detection
#------------------------------------------------------------------------------

def _plot(sd, traces, spike_samples, masks):  # pragma: no cover
    from vispy.app import run
    from phy.plot import plot_traces
    plot_traces(sd.subset_traces(traces),
                spike_samples=spike_samples,
                masks=masks,
                n_samples_per_spike=40)
    run()


class TestConcat(object):
    # [ *  *  0  1  2  3  4  *  * | *  *  5  6  7  8  9  *  *  | *  *  10  11 ]
    # [ !        !        !               !              !                    ]
    # spike_samples: 1, 4, 5

    def setup(self):
        from dask.array import Array, from_array

        self.trace_chunks = ((5, 5, 2), (3,))
        self.depth = 2

        # Create the chunked spike_samples array.
        dask = {('spike_samples', 0): np.array([0, 3, 6]),
                ('spike_samples', 1): np.array([2, 7]),
                ('spike_samples', 2): np.array([]),
                }
        spikes_chunks = ((3, 2, 0),)
        s = Array(dask, 'spike_samples', spikes_chunks,
                  shape=(5,), dtype=np.int32)
        self.spike_samples = s
        # Indices of the spikes that are kept (outside of overlapping bands).
        self.spike_indices = np.array([1, 2, 3])

        assert len(self.spike_samples.compute()) == 5

        self.masks = from_array(np.arange(5 * 3).reshape((5, 3)),
                                spikes_chunks + (3,))
        self.waveforms = from_array(np.arange(5 * 3 * 2).reshape((5, 3, 2)),
                                    spikes_chunks + (3, 2))

    def test_spikes_to_keep(self):
        indices = _spikes_to_keep(self.spike_samples,
                                  self.trace_chunks,
                                  self.depth)
        onsets, offsets = indices
        assert list(zip(onsets, offsets)) == [(1, 3), (0, 1), (0, 0)]

    def test_trim_spikes(self):
        indices = _spikes_to_keep(self.spike_samples,
                                  self.trace_chunks,
                                  self.depth)

        # Trim the spikes.
        spikes_trimmed = _trim_spikes(self.spike_samples, indices)
        ae(spikes_trimmed.compute(), [3, 6, 2])

    def test_add_chunk_offset(self):
        indices = _spikes_to_keep(self.spike_samples,
                                  self.trace_chunks,
                                  self.depth)
        spikes_trimmed = _trim_spikes(self.spike_samples, indices)

        # Add the chunk offsets to the spike samples.
        self.spikes = _add_chunk_offset(spikes_trimmed,
                                        self.trace_chunks, self.depth)
        ae(self.spikes, [1, 4, 5])

    def test_concat(self):
        sc, mc, wc = _concat_spikes(self.spike_samples,
                                    self.masks,
                                    self.waveforms,
                                    trace_chunks=self.trace_chunks,
                                    depth=self.depth,
                                    )
        sc = sc.compute()
        mc = mc.compute()
        wc = wc.compute()

        ae(sc, [1, 4, 5])
        ae(mc, self.masks.compute()[self.spike_indices])
        ae(wc, self.waveforms.compute()[self.spike_indices])


def test_detect_simple(spike_detector, traces):
    sd = spike_detector

    spike_samples, masks, _ = sd.detect(traces)

    n_channels = sd.n_channels
    n_spikes = len(spike_samples)

    assert spike_samples.dtype == np.int64
    assert spike_samples.ndim == 1

    assert masks.dtype == np.float32
    assert masks.ndim == 2
    assert masks.shape == (n_spikes, n_channels)

    # _plot(sd, traces, spike_samples, masks)


def test_detect_context(spike_detector, traces, context, ipy_client):
    sd = spike_detector
    sd.set_context(context)
    context.ipy_view = ipy_client[:]

    spike_samples, masks, _ = sd.detect(traces)

    n_channels = sd.n_channels
    n_spikes = len(spike_samples)

    assert spike_samples.dtype == np.int64
    assert spike_samples.ndim == 1

    assert masks.dtype == np.float32
    assert masks.ndim == 2
    assert masks.shape == (n_spikes, n_channels)
    # _plot(sd, traces, spike_samples.compute(), masks.compute())
