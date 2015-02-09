# -*- coding: utf-8 -*-

"""Mock Kwik files."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os
import os.path as op
from random import randint

import numpy as np
from numpy.testing import assert_array_equal as ae
import h5py
from pytest import raises

from ...io.mock.artificial import (artificial_spike_times,
                                   artificial_spike_clusters,
                                   artificial_features,
                                   artificial_masks,
                                   artificial_traces)
from ...electrode.mea import MEA, staggered_positions
from ...utils.tempdir import TemporaryDirectory
from ..h5 import open_h5
from ..kwik_model import (KwikModel, _list_channel_groups, _list_channels,
                          _list_recordings,
                          _list_clusterings, _kwik_filenames)


#------------------------------------------------------------------------------
# Mock Kwik file
#------------------------------------------------------------------------------

def create_mock_kwik(dir_path, n_clusters=None, n_spikes=None,
                     n_channels=None, n_features_per_channel=None,
                     n_samples_traces=None,
                     with_kwx=True, with_kwd=True):
    """Create a test kwik file."""
    filename = op.join(dir_path, '_test.kwik')
    filenames = _kwik_filenames(filename)
    kwx_filename = filenames['kwx']
    kwd_filename = filenames['raw.kwd']

    # Create the kwik file.
    with open_h5(filename, 'w') as f:
        f.write_attr('/', 'kwik_version', 2)

        def _write_metadata(key, value):
            f.write_attr('/application_data/spikedetekt', key, value)

        _write_metadata('sample_rate', 20000.)

        # Filter parameters.
        _write_metadata('filter_low', 500.)
        _write_metadata('filter_high', 0.95 * .5 * 20000.)
        _write_metadata('filter_butter_order', 3)

        _write_metadata('extract_s_before', 15)
        _write_metadata('extract_s_after', 25)

        _write_metadata('nfeatures_per_channel', n_features_per_channel)

        # Create spike times.
        spike_times = artificial_spike_times(n_spikes).astype(np.int64)

        if spike_times.max() >= n_samples_traces:
            raise ValueError("There are too many spikes: decrease 'n_spikes'.")

        f.write('/channel_groups/1/spikes/time_samples', spike_times)

        # Create spike clusters.
        spike_clusters = artificial_spike_clusters(n_spikes,
                                                   n_clusters).astype(np.int32)
        f.write('/channel_groups/1/spikes/clusters/main', spike_clusters)

        # Create channels.
        positions = staggered_positions(n_channels)
        for channel in range(n_channels):
            group = '/channel_groups/1/channels/{0:d}'.format(channel)
            f.write_attr(group, 'name', str(channel))
            f.write_attr(group, 'position', positions[channel])

        # Create cluster metadata.
        for cluster in range(n_clusters):
            group = '/channel_groups/1/clusters/main/{0:d}'.format(cluster)
            color = ('/channel_groups/1/clusters/main/{0:d}'.format(cluster) +
                     '/application_data/klustaviewa')
            f.write_attr(group, 'cluster_group', 3)
            f.write_attr(color, 'color', randint(2, 10))

        # Create recordings.
        f.write_attr('/recordings/0', 'name', 'recording_0')

    # Create the kwx file.
    if with_kwx:
        with open_h5(kwx_filename, 'w') as f:
            f.write_attr('/', 'kwik_version', 2)
            features = artificial_features(n_spikes,
                                           n_channels * n_features_per_channel)
            masks = artificial_masks(n_spikes,
                                     n_channels * n_features_per_channel)
            fm = np.dstack((features, masks)).astype(np.float32)
            f.write('/channel_groups/1/features_masks', fm)

    # Create the raw kwd file.
    if with_kwd:
        with open_h5(kwd_filename, 'w') as f:
            f.write_attr('/', 'kwik_version', 2)
            traces = artificial_traces(n_samples_traces, n_channels)
            # TODO: int16 traces
            f.write('/recordings/0/data', traces.astype(np.float32))

    return filename
