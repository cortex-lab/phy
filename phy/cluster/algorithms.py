# -*- coding: utf-8 -*-

"""Automatic clustering algorithms."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ..utils.array import PartialArray, get_excerpts
from ..io.kwik.sparse_kk2 import sparsify_features_masks
from ..traces import (Filter, Thresholder, compute_threshold,
                      FloodFillDetector, WaveformExtractor, PCA,
                      )


#------------------------------------------------------------------------------
# Spike detection class
#------------------------------------------------------------------------------

class SpikeDetekt(object):
    def __init__(self, **kwargs):
        self._kwargs = kwargs

        # Data chunks.
        # chunk_size = self._kwargs['chunk_size']
        # chunk_overlap = self._kwargs['chunk_overlap']

    # Processing objects creation
    # -------------------------------------------------------------------------

    def _create_filter(self):
        rate = self._kwargs['sample_rate']
        low = self._kwargs['filter_low']
        high = self._kwargs['filter_high']
        order = self._kwargs['filter_butter_order']
        return Filter(rate=rate,
                      low=low,
                      high=high,
                      order=order,
                      )

    def _create_thresholder(self, thresholds):
        mode = self._kwargs['detect_spikes']
        return Thresholder(mode=mode, thresholds=thresholds)

    def _create_detector(self):
        graph = self._kwargs['probe_adjacency_list']
        join_size = self._kwargs['connected_component_join_size']
        return FloodFillDetector(probe_adjacency_list=graph,
                                 join_size=join_size,
                                 )

    def _create_extractor(self):
        before = self._kwargs['extract_s_before']
        after = self._kwargs['extract_s_after']
        weight_power = self._kwargs['weight_power']
        probe_channels = self._kwargs['probe_channels']
        return WaveformExtractor(extract_before=before,
                                 extract_after=after,
                                 weight_power=weight_power,
                                 channels_per_group=probe_channels,
                                 )

    def _create_pca(self):
        n_pcs = self._kwargs['nfeatures_per_channel']
        return PCA(n_pcs=n_pcs)

    # Processing functions
    # -------------------------------------------------------------------------

    def update_params(self, **kwargs):
        self._kwargs.update(kwargs)

    def apply_filter(self, data):
        return self.filter(data)

    def find_thresholds(self, traces_f):
        """Find weak and strong thresholds in filtered traces."""
        n_excerpts = self._kwargs['nexcerpts']
        excerpt_size = self._kwargs['excerpt_size']
        single = self._kwargs['use_single_threshold']
        strong_f = self._kwargs['threshold_strong_std_factor']
        weak_f = self._kwargs['threshold_weak_std_factor']
        excerpt = get_excerpts(traces_f,
                               n_excerpts=n_excerpts,
                               excerpt_size=excerpt_size)
        return compute_threshold(excerpt,
                                 single_threshold=single,
                                 std_factor=(weak_f, strong_f))

    def detect(self, traces_f, thresholds=None):
        """Detect connected waveform components in filtered traces.

        Parameters
        ----------

        traces_f : array
            An `(n_samples, n_channels)` array with the filtered data.
        thresholds : dict
            The weak and strong thresholds.

        Returns
        -------

        traces_t : array
            An `(n_samples, n_channels)` array with the transformed data
            according to the detection mode.
        components : list
            A list of `(n, 2)` arrays with `sample, channel` pairs.

        """
        # Threshold the data following the weak and strong thresholds.
        thresholder = self._create_thresholder(thresholds)
        # Transform the filtered data according to the detection mode.
        traces_t = thresholder.transform(traces_f)
        # Compute the threshold crossings.
        weak = thresholder.detect(traces_t, 'weak')
        strong = thresholder.detect(traces_t, 'strong')
        return traces_t, self.detector(weak_crossings=weak,
                                       strong_crossings=strong)

    def extract_spikes(self, components, traces_f, traces_t):
        """Extract spikes from connected components.

        Parameters
        ----------
        components : list
            List of connected components.
        traces_f : array
            Filtered data.
        traces_t : array
            Transformed data.

        Returns
        -------

        spike_samples : array
            An `(n_spikes,)` array with the spike samples.
        waveforms : array
            An `(n_spikes, n_samples, n_channels)` array.
        masks : array
            An `(n_spikes, n_channels)` array.

        """
        n_spikes = len(components)
        extractor = self._create_extractor()
        # Extract all waveforms.
        samples, waveforms, masks = zip(*(extractor(component,
                                                    data=traces_f,
                                                    data_t=traces_t)
                                          for component in components))
        # Create the return arrays.
        samples = np.array(samples, dtype=np.uint64)
        waveforms = np.array(waveforms, dtype=np.float32)
        masks = np.array(masks, dtype=np.float32)

        assert samples.shape == (n_spikes,)
        assert waveforms.ndim == 3
        assert waveforms.shape[0] == n_spikes
        _, n_samples, n_channels = waveforms.shape
        assert masks.shape == (n_spikes, n_channels)

        return samples, waveforms, masks

    def features(self, waveforms, pcs):
        """Extract features from waveforms.

        Returns
        -------

        features : array
            An `(n_spikes, n_channels, n_features)` array.

        """
        n_waveforms_max = self._kwargs['pca_nwaveforms_max']

    # Main function
    # -------------------------------------------------------------------------

    def run(self, traces, interval=None):

        # Filter the traces.
        traces_f = self.apply_filter(traces)


#------------------------------------------------------------------------------
# Clustering class
#------------------------------------------------------------------------------

class KlustaKwik(object):
    """KlustaKwik automatic clustering algorithm."""
    def __init__(self, **kwargs):
        assert 'num_starting_clusters' in kwargs
        self._kwargs = kwargs
        self.__dict__.update(kwargs)
        # Set the version.
        from klustakwik2 import __version__
        self.version = __version__

    def cluster(self,
                model=None,
                spike_ids=None,
                features=None,
                masks=None,
                ):
        # Get the features and masks.
        if model is not None:
            if features is None:
                features = PartialArray(model.features_masks, 0)
            if masks is None:
                masks = PartialArray(model.features_masks, 1)
        # Select some spikes if needed.
        if spike_ids is not None:
            features = features[spike_ids]
            masks = masks[spike_ids]
        # Convert the features and masks to the sparse structure used
        # by KK.
        data = sparsify_features_masks(features, masks)
        data = data.to_sparse_data()
        # Run KK.
        from klustakwik2 import KK
        num_starting_clusters = self._kwargs.pop('num_starting_clusters', 100)
        kk = KK(data, **self._kwargs)
        self.params = kk.all_params
        self.params['num_starting_clusters'] = num_starting_clusters
        kk.cluster_mask_starts(num_starting_clusters)
        spike_clusters = kk.clusters
        return spike_clusters


def cluster(model, algorithm='klustakwik', spike_ids=None, **kwargs):
    """Launch an automatic clustering algorithm on the model.

    Parameters
    ----------

    model : BaseModel
        A model.
    algorithm : str
        Only 'klustakwik' is supported currently.
    **kwargs
        Parameters for KK.

    """
    assert algorithm == 'klustakwik'
    kk = KlustaKwik(**kwargs)
    return kk.cluster(model=model, spike_ids=spike_ids)
