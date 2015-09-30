# -*- coding: utf-8 -*-

"""Tests of spike detection routines."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from pytest import yield_fixture

from phy.utils.datasets import download_test_data
from phy.utils.tests.test_context import context, ipy_client
from phy.electrode import load_probe
from ..spike_detect import (SpikeDetector, _concat_spikes,
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


def test_detect_concat():
    import dask.async
    from dask import set_options
    from dask.array import Array, from_array
    set_options(get=dask.async.get_sync)

    chunks = ((5, 5, 2), (3,))
    depth = 2
    # [ 0  1  2  3  4 | 5  6  7  8  9 | 10  11 ]
    # [ 0     2  3               8  9          ]

    # Traces
    # [ *  *  0  1  2  3  4  *  * | *  *  5  6  7  8  9  *  *  | *  *  10  11 ]
    # [ !        !        !               !              !                    ]
    # Spikes

    dask = {('s', 0): np.array([0, 3, 6]),
            ('s', 1): np.array([2, 7]),
            ('s', 2): np.array([]),
            }
    chunks_spikes = ((3, 2, 0),)
    s = Array(dask, 's', chunks_spikes, shape=(5,), dtype=np.int32)
    m = from_array(np.arange(5 * 3).reshape((5, 3)),
                   chunks_spikes + (3,))
    w = from_array(np.arange(5 * 3 * 2).reshape((5, 3, 2)),
                   chunks_spikes + (3, 2))

    sc, mc, wc = _concat_spikes(s, m, w, chunks=chunks, depth=depth)
    sc = sc.compute()
    mc = mc.compute()
    wc = wc.compute()

    print(sc)
    print(mc)
    print(wc)


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


def test_detect_context(spike_detector, traces, context):
    sd = spike_detector
    sd.set_context(context)
    # context.ipy_view = ipy_client[:]

    from dask.array import from_array
    traces_da = from_array(traces, chunks=(5000, traces.shape[1]))
    spike_samples, masks, _ = sd.detect(traces_da)

    n_channels = sd.n_channels
    n_spikes = len(spike_samples)

    assert spike_samples.dtype == np.int64
    assert spike_samples.ndim == 1

    assert masks.dtype == np.float32
    assert masks.ndim == 2
    assert masks.shape == (n_spikes, n_channels)
    # _plot(sd, traces, spike_samples.compute(), masks.compute())
