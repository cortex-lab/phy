"""Tests for the example reclustering plugin."""

from types import SimpleNamespace

import numpy as np

from plugins import recluster


def test_recluster_splits_selected_spikes(monkeypatch):
    spike_ids = np.arange(4)
    split_calls = []
    controller = SimpleNamespace(
        model=SimpleNamespace(features=object(), features_rows=None),
        supervisor=SimpleNamespace(
            selected=[7],
            clustering=SimpleNamespace(spikes_in_clusters=lambda cluster_ids: spike_ids),
            actions=SimpleNamespace(
                split=lambda selected_spikes, labels: split_calls.append((selected_spikes, labels))
            ),
        ),
        get_best_channels=lambda cluster_id: np.array([2, 3]),
        _get_spike_features=lambda selected_spikes, channel_ids: SimpleNamespace(
            data=np.arange(16, dtype=float).reshape(4, 2, 2)
        ),
    )
    monkeypatch.setattr(recluster, '_reduce', lambda x: x)
    monkeypatch.setattr(
        recluster, '_isosplit', lambda x, dip_threshold=None: np.array([0, 0, 1, 1])
    )

    recluster._recluster(controller)

    assert len(split_calls) == 1
    np.testing.assert_array_equal(split_calls[0][0], spike_ids)
    np.testing.assert_array_equal(split_calls[0][1], [0, 0, 1, 1])
