# -*- coding: utf-8 -*-

"""Automatic clustering algorithms."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op
from collections import defaultdict

import numpy as np

from ..utils.array import (PartialArray, get_excerpts,
                           chunk_bounds, data_chunk,
                           _load_ndarray, _as_array,
                           )
from ..utils._types import Bunch
from ..utils.event import EventEmitter
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


def _split_spikes(groups, idx=None, **arrs):
    """Split spike data according to the channel group."""
    groups = _as_array(groups)
    if idx is not None:
        n_spikes_chunk = np.sum(idx)
        # First, remove the overlapping bands.
        groups = groups[idx]
        arrs_bis = arrs.copy()
        for key, arr in arrs.items():
            arrs_bis[key] = arr[idx, ...]
            assert len(arrs_bis[key]) == n_spikes_chunk
    # Then, split along the group.
    groups_u = np.unique(groups)
    out = {}
    for group in groups_u:
        i = (groups == group)
        out[group] = {}
        for key, arr in arrs_bis.items():
            out[group][key] = arr[i, ...]
    return out


#------------------------------------------------------------------------------
# Spike detection class
#------------------------------------------------------------------------------

class SpikeDetekt(EventEmitter):
    def __init__(self, tempdir=None, **kwargs):
        super(SpikeDetekt, self).__init__()
        self._tempdir = tempdir
        self._kwargs = kwargs
        self._n_channels_per_group = {
            group: len(channels)
            for group, channels in self._kwargs['probe_channels'].items()
        }
        self._n_features = self._kwargs['nfeatures_per_channel']

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

    def _create_extractor(self, thresholds):
        before = self._kwargs['extract_s_before']
        after = self._kwargs['extract_s_after']
        weight_power = self._kwargs['weight_power']
        probe_channels = self._kwargs['probe_channels']
        self._n_samples_waveforms = before + after
        return WaveformExtractor(extract_before=before,
                                 extract_after=after,
                                 weight_power=weight_power,
                                 channels_per_group=probe_channels,
                                 thresholds=thresholds,
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
        return filter(data).astype(np.float32)

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
        thresholds = compute_threshold(excerpt_f,
                                       single_threshold=single,
                                       std_factor=(weak_f, strong_f))
        return {'weak': thresholds[0],
                'strong': thresholds[1]}

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

    def extract_spikes(self, components, traces_f, thresholds=None):
        """Extract spikes from connected components.

        Parameters
        ----------
        components : list
            List of connected components.
        traces_f : array
            Filtered data.
        thresholds : dict
            The weak and strong thresholds.

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
        assert n_spikes > 0
        # Transform the filtered data according to the detection mode.
        thresholder = self._create_thresholder()
        traces_t = thresholder.transform(traces_f)
        # Extract all waveforms.
        extractor = self._create_extractor(thresholds)
        groups, samples, waveforms, masks = zip(*[extractor(component,
                                                            data=traces_f,
                                                            data_t=traces_t,
                                                            )
                                                  for component in components])
        # Create the return arrays.
        groups = np.array(groups)
        samples = np.array(samples, dtype=np.uint64)
        waveforms = np.array(waveforms, dtype=np.float32)
        masks = np.array(masks, dtype=np.float32)

        # Reorder the spikes.
        idx = np.argsort(samples)
        groups = groups[idx].astype(np.int32)
        samples = samples[idx]
        waveforms = waveforms[idx, ...]
        masks = masks[idx, ...]

        assert groups.shape == (n_spikes,)
        assert samples.shape == (n_spikes,)
        assert waveforms.ndim == 3
        assert waveforms.shape[0] == n_spikes
        _, n_samples, n_channels = waveforms.shape
        assert masks.shape == (n_spikes, n_channels)

        assert groups.dtype == np.int32
        assert samples.dtype == np.uint64
        assert waveforms.dtype == np.float32
        assert masks.dtype == np.float32

        return groups, samples, waveforms, masks

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

    # Internal functions
    # -------------------------------------------------------------------------

    def _path(self, name, key=None, group=None):
        if self._tempdir is None:
            raise ValueError("The temporary directory must be specified.")
        assert key >= 0
        if group is None:
            path = op.join(self._tempdir, '{:s}-{:d}'.format(name, key))
        else:
            assert group >= 0
            fn = '{chunk:d}.{name:s}.{group:d}'.format(
                chunk=key, name=name, group=group)
            path = op.join(self._tempdir, fn)
        return path

    def _save(self, array, name, key=None, group=None):
        path = self._path(name, key=key, group=group)
        dtype = array.dtype
        shape = array.shape
        debug("Save `{}` ({}, {}).".format(path, np.dtype(dtype).name, shape))
        return array.tofile(path)

    def _load(self, name, dtype, shape=None, key=None, group=None):
        path = self._path(name, key=key, group=group)
        assert op.exists(path)
        debug("Load `{}` ({}, {}).".format(path, np.dtype(dtype).name, shape))
        with open(path, 'rb') as f:
            return _load_ndarray(f, dtype=dtype, shape=shape)

    def _load_data_chunks(self, name, dtype,
                          n_samples=None,
                          n_channels=None,
                          groups=None,
                          n_spikes_per_group_chunk=None,
                          ):
        _, _, keys, _ = zip(*list(self.iter_chunks(n_samples, n_channels)))
        out = {}
        for group in groups:
            out[group] = []
            n_channels_group = self._n_channels_per_group[group]
            for key in keys:
                n_spikes = n_spikes_per_group_chunk[group][key]
                shape = {
                    'waveforms': (n_spikes,
                                  self._n_samples_waveforms,
                                  n_channels_group),
                    'masks': (n_spikes, n_channels_group),
                    'features': (n_spikes, n_channels_group, self._n_features),
                }[name]
                w = self._load(name, dtype,
                               shape=shape,
                               key=key,
                               group=group)
                out[group].append(w)
        return out

    def _pca_subset(self, wm, n_spikes_chunk=None, n_spikes_total=None):
        waveforms, masks = wm
        n_waveforms_max = self._kwargs['pca_nwaveforms_max']
        p = n_spikes_chunk / float(n_spikes_total)
        k = int(n_spikes_chunk / float(p * n_waveforms_max))
        k = np.clip(k, 1, n_spikes_chunk)
        return (waveforms[::k, ...], masks[::k, ...])

    def iter_chunks(self, n_samples, n_channels):
        chunk_size = self._kwargs['chunk_size']
        overlap = self._kwargs['chunk_overlap']
        for bounds in chunk_bounds(n_samples, chunk_size, overlap=overlap):
            yield bounds

    def output_data(self,
                    n_samples,
                    n_channels,
                    groups=None,
                    n_spikes_per_group_chunk=None,
                    ):
        n_samples_per_chunk = {bounds[2]: (bounds[3] - bounds[2])
                               for bounds in self.iter_chunks(n_samples,
                                                              n_channels)}
        keys = sorted(n_samples_per_chunk.keys())
        traces_f = [self._load('filtered', np.float32,
                               shape=(n_samples_per_chunk[key], n_channels),
                               key=key) for key in keys]

        def _load(name):
            nspgc = n_spikes_per_group_chunk
            return self._load_data_chunks(name, np.float32,
                                          n_samples=n_samples,
                                          n_channels=n_channels,
                                          groups=groups,
                                          n_spikes_per_group_chunk=nspgc,
                                          )

        output = Bunch(chunk_keys=keys,
                       traces_f=traces_f,
                       waveforms=_load('waveforms'),
                       masks=_load('masks'),
                       features=_load('features'),
                       )
        return output

    # Main steps
    # -------------------------------------------------------------------------

    def step_detect(self, bounds, chunk_data, chunk_data_keep,
                    thresholds=None):
        key = bounds[2]
        # Apply the filter.
        data_f = self.apply_filter(chunk_data)
        assert data_f.dtype == np.float32
        assert data_f.shape == chunk_data.shape
        # Save the filtered chunk.
        self._save(data_f, 'filtered', key=key)
        # Detect spikes in the filtered chunk.
        components = self.detect(data_f, thresholds=thresholds)
        # Return the list of components in the chunk.
        return components

    def step_extract(self, bounds, components,
                     n_spikes_total=None,
                     n_channels=None,
                     thresholds=None,
                     ):
        """Return the waveforms to keep for each chunk for PCA."""
        assert len(components) > 0
        s_start, s_end, keep_start, keep_end = bounds
        key = keep_start
        n_samples = s_end - s_start
        # Get the filtered chunk.
        chunk_f = self._load('filtered', np.float32,
                             shape=(n_samples, n_channels), key=key)
        # Extract the spikes from the chunk.
        groups, spike_samples, waveforms, masks = self.extract_spikes(
            components, chunk_f, thresholds=thresholds)

        # Remove spikes in the overlapping bands.
        idx = _keep_spikes(spike_samples, (keep_start, keep_end))
        n_spikes_chunk = len(idx)
        # Split the data according to the channel groups.
        split = _split_spikes(groups,
                              idx=idx,
                              spike_samples=spike_samples,
                              waveforms=waveforms,
                              masks=masks,
                              )
        # Save the split arrays: spike samples, waveforms, masks.
        for group, out in split.items():
            for name, arr in out.items():
                self._save(arr, name, key=key, group=group)

        # Keep some waveforms in memory in order to compute PCA.
        wm = {group: (split[group]['waveforms'], split[group]['masks'])
              for group in split.keys()}
        wm = {group: self._pca_subset(wm[group],
                                      n_spikes_chunk=n_spikes_chunk,
                                      n_spikes_total=n_spikes_total)
              for group in split.keys()}
        return wm

    def step_pca(self, chunk_waveforms):
        # This is a dict {key: (waveforms, masks)}.
        # Concatenate all waveforms subsets from all chunks.
        waveforms_subset, masks_subset = zip(*chunk_waveforms.values())
        waveforms_subset = np.vstack(waveforms_subset)
        masks_subset = np.vstack(masks_subset)
        assert (waveforms_subset.shape[0],
                waveforms_subset.shape[2]) == masks_subset.shape
        # Perform PCA and return the components.
        pcs = self.waveform_pcs(waveforms_subset, masks_subset)
        return pcs

    def step_features(self, bounds, pcs_per_group, n_spikes_per_group_chunk):
        s_start, s_end, keep_start, keep_end = bounds
        key = keep_start
        # Loop over the channel groups.
        for group, pcs in pcs_per_group.items():
            # Find the waveforms shape.
            n_spikes = n_spikes_per_group_chunk[group][key]
            n_channels = self._n_channels_per_group[group]
            shape = (n_spikes, self._n_samples_waveforms, n_channels)
            # Save the waveforms.
            waveforms = self._load('waveforms', np.float32,
                                   shape=shape,
                                   key=key, group=group)
            # Compute the features.
            features = self.features(waveforms, pcs)
            assert features.dtype == np.float32
            # Save the features.
            self._save(features, 'features', key=key, group=group)

    def run_serial(self, traces, interval=None):
        """Run SpikeDetekt using one CPU."""
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
        self.emit('find_thresholds', thresholds)
        debug("Thresholds: {}.".format(thresholds))

        # Pass 1: find the connected components and count the spikes.
        info("Detecting spikes...")
        # Dictionary {chunk_key: components}.
        # Every chunk has a unique key: the `keep_start` integer.
        chunk_components = {}
        for bounds in self.iter_chunks(n_samples, n_channels):
            key = bounds[2]
            chunk_data = data_chunk(traces, bounds, with_overlap=True)
            chunk_data_keep = data_chunk(traces, bounds, with_overlap=False)
            components = self.step_detect(bounds,
                                          chunk_data,
                                          chunk_data_keep,
                                          thresholds=thresholds,
                                          )
            self.emit('detect_spikes', key=key, n_spikes=len(components))
            chunk_components[key] = components
        n_spikes_per_chunk = {key: len(val)
                              for key, val in chunk_components.items()}
        n_spikes_total = sum(n_spikes_per_chunk.values())
        info("{} spikes detected in total.".format(n_spikes_total))

        # Pass 2: extract the spikes and save some waveforms before PCA.
        info("Extracting all waveforms...")
        # This is a dict {group: {key: (waveforms, masks)}}.
        chunk_waveforms = defaultdict(dict)
        for bounds in self.iter_chunks(n_samples, n_channels):
            key = bounds[2]
            components = chunk_components[key]
            if len(components) == 0:
                continue
            # This is a dict {group: (waveforms, masks)}.
            wm = self.step_extract(bounds,
                                   components,
                                   n_spikes_total=n_spikes_total,
                                   n_channels=n_channels,
                                   thresholds=thresholds,
                                   )
            self.emit('extract_spikes', key=key, n_spikes=len(components))
            for group, wm_group in wm.items():
                chunk_waveforms[group][key] = wm_group
        n_spikes_per_group_chunk = {
            group: {key: len(val)
                    for key, val in chunk_waveforms[group].items()
                    }
            for group in chunk_waveforms.keys()
        }
        groups = sorted(chunk_waveforms.keys())
        info("All waveforms extracted and saved.")

        # Compute the PCs.
        info("Performing PCA...")
        pcs = {}
        for group in groups:
            pcs[group] = self.step_pca(chunk_waveforms[group])
            self.emit('compute_pca', group=group, pcs=pcs[group])
        info("Principal waveform components computed.")

        # Pass 3: compute the features.
        info("Computing the features of all spikes...")
        for bounds in self.iter_chunks(n_samples, n_channels):
            self.step_features(bounds, pcs,
                               n_spikes_per_group_chunk)
            self.emit('compute_features', key=bounds[2])
        info("All features computed and saved.")

        # Return dictionary of memmapped data.
        return self.output_data(n_samples, n_channels,
                                groups, n_spikes_per_group_chunk)


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
