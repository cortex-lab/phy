# -*- coding: utf-8 -*-

"""Test GUI component."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from ..store import create_cluster_store, ClusterStore
from phy.io import Context, Selector


#------------------------------------------------------------------------------
# Test cluster stats
#------------------------------------------------------------------------------

def test_cluster_store(tempdir):
    context = Context(tempdir)
    cs = ClusterStore(context=context)

    @cs.add(cache='memory')
    def f(x):
        return x * x

    assert cs.f(3) == 9
    assert cs.f(3) == 9


def test_create_cluster_store(model):
    selector = Selector(spike_clusters=model.spike_clusters,
                        spikes_per_cluster=model.spikes_per_cluster)
    cs = create_cluster_store(model, selector=selector)

    nc = model.n_channels
    nfpc = model.n_features_per_channel
    ns = len(model.spikes_per_cluster[1])
    ns2 = len(model.spikes_per_cluster[2])
    nsw = model.n_samples_waveforms

    def _check(out, *shape):
        spikes, arr = out
        assert spikes.shape[0] == shape[0]
        assert arr.shape == shape

    # Model data.
    _check(cs.masks(1), ns, nc)
    _check(cs.features(1), ns, nc, nfpc)
    _check(cs.waveforms(1), ns, nsw, nc)

    # Waveforms masks.
    spike_ids, w, m = cs.waveforms_masks(1)
    _check((spike_ids, w), ns, nsw, nc)
    _check((spike_ids, m), ns, nc)

    # Background feature masks.
    spike_ids, bgf, bgm = cs.background_features_masks()
    assert bgf.ndim == 3
    assert bgf.shape[1:] == (nc, nfpc)
    assert bgm.ndim == 2
    assert bgm.shape[1] == nc
    assert spike_ids.shape == (bgf.shape[0],) == (bgm.shape[0],)

    # Test concat multiple clusters.
    spike_ids, f, m = cs.features_masks([1, 2])
    assert len(spike_ids) == ns + ns2
    assert f.shape == (ns + ns2, nc, nfpc)
    assert m.shape == (ns + ns2, nc)

    # Test means.
    assert cs.mean_masks(1).shape == (nc,)
    assert cs.mean_features(1).shape == (nc, nfpc)
    assert cs.mean_waveforms(1).shape == (nsw, nc)

    # Limits.
    assert 0 < cs.waveform_lim() < 1
    assert 0 < cs.feature_lim() < 1
    assert cs.mean_traces().shape == (1, nc)

    # Statistics.
    assert 1 <= len(cs.best_channels(1)) <= nc
    assert 1 <= len(cs.best_channels_multiple([1, 2])) <= nc
    assert 0 < cs.max_waveform_amplitude(1) < 1
    assert cs.mean_masked_features_score(1, 2) > 0
