# -*- coding: utf-8 -*-

"""Tests of Kwik file creator."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op

import numpy as np
from numpy.testing import assert_array_equal as ae
from numpy.testing import assert_allclose as ac

from ....utils.tempdir import TemporaryDirectory
from ...h5 import open_h5
from ..creator import KwikCreator, _write_by_chunk
from ..mock import (artificial_spike_samples,
                    artificial_features,
                    artificial_masks,
                    )


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_write_by_chunk():
    n = 5
    arrs = [np.random.rand(i + 1, 3) for i in range(n)]
    n_tot = n * (n + 1) // 2

    with TemporaryDirectory() as tempdir:
        path = op.join(tempdir, 'test.h5')
        with open_h5(path, 'w') as f:
            ds = f.write('/test', shape=(n_tot, 3), dtype=np.float32)
            _write_by_chunk(ds, arrs)
        with open_h5(path, 'r') as f:
            ds = f.read('/test')[...]
            offset = 0
            for i, arr in enumerate(arrs):
                size = arr.shape[0]
                assert size == (i + 1)
                ac(ds[offset:offset + size, ...], arr)
                offset += size


def test_creator():
    with TemporaryDirectory() as tempdir:
        basename = op.join(tempdir, 'my_file')

        creator = KwikCreator(basename)

        # Test create empty files.
        creator.create_empty()
        assert op.exists(basename + '.kwik')
        assert op.exists(basename + '.kwx')

        # Test metadata.
        creator.set_metadata('/application_data/spikedetekt',
                             a=1, b=2., c=[0, 1])

        with open_h5(creator.kwik_path, 'r') as f:
            assert f.read_attr('/application_data/spikedetekt', 'a') == 1
            assert f.read_attr('/application_data/spikedetekt', 'b') == 2.
            ae(f.read_attr('/application_data/spikedetekt', 'c'), [0, 1])

        # Test add spikes.
        n_spikes = 100
        n_channels = 8
        n_features = 3

        spike_samples = artificial_spike_samples(n_spikes)
        features = artificial_features(n_spikes, n_channels, n_features)
        masks = artificial_masks(n_spikes, n_channels)

        creator.add_spikes(group=0,
                           spike_samples=spike_samples,
                           features=features,
                           masks=masks,
                           )
