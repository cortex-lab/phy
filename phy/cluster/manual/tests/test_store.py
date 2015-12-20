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

def test_create_cluster_store(model):
    selector = Selector(spike_clusters=model.spike_clusters,
                        spikes_per_cluster=model.spikes_per_cluster)
    cs = create_cluster_store(model, selector=selector)
    assert cs.mean_masks(1).shape == (model.n_channels,)
    assert cs.mean_features(1).shape == (model.n_channels,
                                         model.n_features_per_channel)
    assert cs.mean_waveforms(1).shape == (model.n_samples_waveforms,
                                          model.n_channels)
    assert 1 <= cs.best_channels(1).shape[0] <= model.n_channels
    assert 0 < cs.max_waveform_amplitude(1) < 1
    assert cs.mean_masked_features_score(1, 2) > 0


def test_cluster_store(tempdir):
    context = Context(tempdir)
    cs = ClusterStore(context=context)

    @cs.add(cache='memory')
    def f(x):
        return x * x

    assert cs.f(3) == 9
    assert cs.f(3) == 9
