# -*- coding: utf-8 -*-

"""Automatic clustering algorithms."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ..utils.array import (PartialArray, get_excerpts,
                           chunk_bounds, data_chunk,
                           )
from ..utils.logging import debug, info
from ..io.kwik.sparse_kk2 import sparsify_features_masks
from ..traces import (Filter, Thresholder, compute_threshold,
                      FloodFillDetector, WaveformExtractor, PCA,
                      )


#------------------------------------------------------------------------------
# Spike detection class
#------------------------------------------------------------------------------

def _keep_spikes(samples, bounds):
    """Only keep spikes within the bounds `bounds=(start, end)`."""
    start, end = bounds
    return (start <= samples) & (samples <= end)


#------------------------------------------------------------------------------
# Spike detection class
#------------------------------------------------------------------------------

class SpikeDetekt(object):
    def __init__(self, **kwargs):
        self._kwargs = kwargs

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

    def _create_thresholder(self, thresholds=None):
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

    # Misc functions
    # -------------------------------------------------------------------------

    def update_params(self, **kwargs):
        self._kwargs.update(kwargs)

    # Processing functions
    # -------------------------------------------------------------------------

    def apply_filter(self, data):
        filter = self._create_filter()
        return filter(data)

    def find_thresholds(self, traces):
        """Find weak and strong thresholds in filtered traces."""
        n_excerpts = self._kwargs['nexcerpts']
        excerpt_size = self._kwargs['excerpt_size']
        single = self._kwargs['use_single_threshold']
        strong_f = self._kwargs['threshold_strong_std_factor']
        weak_f = self._kwargs['threshold_weak_std_factor']
        excerpt = get_excerpts(traces,
                               n_excerpts=n_excerpts,
                               excerpt_size=excerpt_size)
        excerpt_f = self.apply_filter(excerpt)
        return compute_threshold(excerpt_f,
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
        detector = self._create_detector()
        return detector(weak_crossings=weak,
                        strong_crossings=strong)

    def extract_spikes(self, components, traces_f):
        """Extract spikes from connected components.

        Parameters
        ----------
        components : list
            List of connected components.
        traces_f : array
            Filtered data.

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
        # Transform the filtered data according to the detection mode.
        thresholder = self._create_thresholder()
        traces_t = thresholder.transform(traces_f)
        # Extract all waveforms.
        extractor = self._create_extractor()
        samples, waveforms, masks = zip(*(extractor(component,
                                                    data=traces_f,
                                                    data_t=traces_t)
                                          for component in components))
        # Create the return arrays.
        samples = np.array(samples, dtype=np.uint64)
        waveforms = np.array(waveforms, dtype=np.float32)
        masks = np.array(masks, dtype=np.float32)

        # Reorder the spikes.
        idx = np.argsort(samples)
        samples = samples[idx]
        waveforms = waveforms[idx, ...]
        masks = masks[idx, ...]

        assert samples.shape == (n_spikes,)
        assert waveforms.ndim == 3
        assert waveforms.shape[0] == n_spikes
        _, n_samples, n_channels = waveforms.shape
        assert masks.shape == (n_spikes, n_channels)

        return samples, waveforms, masks

    def waveform_pcs(self, waveforms, masks):
        """Compute waveform principal components.

        Returns
        -------

        pcs : array
            An `(n_features, n_samples, n_channels)` array.

        """
        pca = self._create_pca()
        return pca.fit(waveforms, masks)

    def features(self, waveforms, pcs):
        """Extract features from waveforms.

        Returns
        -------

        features : array
            An `(n_spikes, n_channels, n_features)` array.

        """
        pca = self._create_pca()
        return pca.transform(waveforms, pcs=pcs)

    # Main functions
    # -------------------------------------------------------------------------

    def iter_chunks(self, traces):
        n_samples, n_channels = traces.shape

        chunk_size = self._kwargs['chunk_size']
        overlap = self._kwargs['chunk_overlap']

        for bounds in chunk_bounds(n_samples, chunk_size, overlap=overlap):
            chunk = data_chunk(traces, bounds, with_overlap=True)
            # Get the filtered chunk.
            chunk_f = self.apply_filter(chunk)
            yield bounds, chunk, chunk_f

    def run_serial(self, traces, interval=None):
        """Run SpikeDetekt on one core."""
        n_samples, n_channels = traces.shape

        # Take a subset if necessary.
        if interval is not None:
            start, end = interval
            traces = traces[start:end, ...]
        else:
            start, end = 0, n_samples
        assert 0 <= start < end <= n_samples

        # Find the weak and strong thresholds.
        info("Finding the thresholds...")
        thresholds = self.find_thresholds(traces)
        debug("Thresholds: {}.".format(thresholds))

        # PASS 1: find the connected components and count the spikes
        # ----------------------------------------------------------

        # TODO OPTIM: save that on disk instead of in memory.
        # Ditionary {chunk_start: components}.
        chunk_components = {}
        info("Pass 1: detect spikes...")
        for bounds, chunk, chunk_f in self.iter_chunks(traces):
            s_start, s_end, keep_start, keep_end = bounds

            # Detect the connected components in the chunk.
            components = self.detect(chunk_f, thresholds=thresholds)
            chunk_components[s_start] = components

        # Count the total number of spikes.
        n_spikes_per_chunk = {key: len(val)
                              for key, val in chunk_components.items()}
        n_spikes_total = sum(n_spikes_per_chunk.values())
        info("{} spikes detected in total.".format(n_spikes_total))

        # PASS 2: extract the spikes and save some waveforms before PCA
        # -------------------------------------------------------------

        # Waveforms to keep for each chunk in order to compute the PCs.
        chunk_waveforms = {}
        n_waveforms_max = self._kwargs['pca_nwaveforms_max']
        for bounds, chunk, chunk_f in self.iter_chunks(traces):
            s_start, s_end, keep_start, keep_end = bounds

            # Get the previously-computed component list.
            components = chunk_components[s_start]

            # Extract the spikes from the chunk.
            spike_samples, waveforms, masks = self.extract_spikes(components,
                                                                  chunk_f,
                                                                  )

            # Remove spikes in the overlapping bands.
            idx = _keep_spikes(spike_samples, bounds[2:])
            spike_samples = spike_samples[idx]
            waveforms = waveforms[idx, ...]
            masks = masks[idx, ...]

            # TODO: save spikes, masks, waveforms on disk

            # Keep some waveforms in memory in order to compute PCA.
            n_spikes_chunk = len(components)
            # What fraction of all spikes are in that chunk?
            p = n_spikes_chunk / float(n_spikes_total)
            k = int(n_spikes_chunk / float(p * n_waveforms_max))
            k = np.clip(k, 1, n_spikes_chunk)
            w = waveforms[::k, ...]
            m = masks[::k, ...]
            chunk_waveforms[s_start] = w, m

        # Compute the PCs.
        waveforms_subset, masks_subset = zip(*chunk_waveforms.values())
        waveforms_subset = np.array(waveforms_subset)
        masks_subset = np.array(masks_subset)
        assert (waveforms.shape[0], waveforms.shape[2]) == masks_subset.shape
        pcs = self.waveform_pcs(waveforms_subset, masks_subset)

        # PASS 3: compute the features
        # ----------------------------

        for bounds, chunk, chunk_f in self.iter_chunks(traces):
            s_start, s_end, keep_start, keep_end = bounds
            # TODO: load waveforms from disk
            waveforms = None
            self.features(waveforms, pcs)


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
