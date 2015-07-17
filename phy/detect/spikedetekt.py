# -*- coding: utf-8 -*-

"""Automatic clustering algorithms."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os
import os.path as op
from collections import defaultdict

import numpy as np

from ..utils.array import (get_excerpts,
                           chunk_bounds,
                           data_chunk,
                           _as_array,
                           _save_arrays,
                           _load_arrays,
                           )
from ..utils._types import Bunch
from ..utils.event import EventEmitter, ProgressReporter
from ..utils.logging import debug, info
from ..utils.settings import _ensure_dir_exists
from ..electrode.mea import (_channels_per_group,
                             _probe_adjacency_list,
                             )
from ..traces import (Filter, Thresholder, compute_threshold,
                      FloodFillDetector, WaveformExtractor, PCA,
                      )


#------------------------------------------------------------------------------
# Spike detection class
#------------------------------------------------------------------------------

def _find_dead_channels(channels_per_group, n_channels):
    all_channels = sorted([item for sublist in channels_per_group.values()
                           for item in sublist])
    dead = np.setdiff1d(np.arange(n_channels), all_channels)
    debug("Using dead channels: {}.".format(dead))
    return dead


def _keep_spikes(samples, bounds):
    """Only keep spikes within the bounds `bounds=(start, end)`."""
    start, end = bounds
    return (start <= samples) & (samples <= end)


def _split_spikes(groups, idx=None, **arrs):
    """Split spike data according to the channel group."""
    # split: {group: {'spike_samples': ..., 'waveforms':, 'masks':}}
    dtypes = {'spike_samples': np.float64,
              'waveforms': np.float32,
              'masks': np.float32,
              }
    groups = _as_array(groups)
    if idx is not None:
        n_spikes_chunk = np.sum(idx)
        # First, remove the overlapping bands.
        groups = groups[idx]
        arrs_bis = arrs.copy()
        for key, arr in arrs.items():
            arrs_bis[key] = arr[idx]
            assert len(arrs_bis[key]) == n_spikes_chunk
    # Then, split along the group.
    groups_u = np.unique(groups)
    out = {}
    for group in groups_u:
        i = (groups == group)
        out[group] = {}
        for key, arr in arrs_bis.items():
            out[group][key] = _concat(arr[i], dtypes.get(key, None))
    return out


def _array_list(arrs):
    out = np.empty((len(arrs),), dtype=np.object)
    out[:] = arrs
    return out


def _concat(arr, dtype=None):
    out = np.array([_[...] for _ in arr], dtype=dtype)
    return out


def _concatenate(arrs, shape):
    arrs = [arr for arr in arrs if arr is not None]
    if not arrs:
        return np.zeros((0,) + shape)
    return np.concatenate(arrs, axis=0)


def _cut_traces(traces, interval_samples):
    n_samples, n_channels = traces.shape
    # Take a subset if necessary.
    if interval_samples is not None:
        start, end = interval_samples
        assert start <= end
        traces = traces[start:end, ...]
        n_samples = traces.shape[0]
    else:
        start, end = 0, n_samples
    assert 0 <= start < n_samples
    if start > 0:
        # TODO: add offset to the spike samples...
        raise NotImplementedError("Need to add `start` to the "
                                  "spike samples")
    return traces, start


class SpikeCounts(object):
    """Count spikes in chunks and channel groups."""
    def __init__(self, counts):
        self._counts = counts
        self._groups = sorted(counts)

    @property
    def counts(self):
        return self._counts

    def per_group(self, group):
        return sum(self._counts.get(group, {}).values())

    def per_chunk(self, chunk):
        return sum(self._counts[group].get(chunk, 0) for group in self._groups)

    def __call__(self, group=None, chunk=None):
        if group is not None and chunk is not None:
            return self._counts.get(group, {}).get(chunk, 0)
        elif group is not None:
            return self.per_group(group)
        elif chunk is not None:
            return self.per_chunk(chunk)
        elif group is None and chunk is None:
            return sum(self.per_group(group) for group in self._groups)


#------------------------------------------------------------------------------
# Spike detection class
#------------------------------------------------------------------------------

_spikes_message = "{n_spikes:d} spikes in chunk {value:d}/{value_max:d}."


class SpikeDetektProgress(ProgressReporter):
    _progress_messages = {
        'detect': ("Detecting spikes: {progress:.2f}%. " + _spikes_message,
                   "Spike detection complete: {n_spikes_total:d} " +
                   "spikes detected."),

        'excerpt': ("Extracting waveforms subset for PCs: " +
                    "{progress:.2f}%. " + _spikes_message,
                    "Waveform subset extraction complete: " +
                    "{n_spikes_total} spikes."),

        'pca': ("Performing PCA: {progress:.2f}%.",
                "Principal waveform components computed."),

        'extract': ("Extracting spikes: {progress:.2f}%. ",
                    "Spike extraction complete: {n_spikes_total:d} " +
                    "spikes extracted."),

    }

    def __init__(self, n_chunks=None):
        super(SpikeDetektProgress, self).__init__()
        self.n_chunks = n_chunks

    def start_step(self, name, value_max):
        self._iter = 0
        self.reset(value_max)
        self.set_progress_message(self._progress_messages[name][0],
                                  line_break=True)
        self.set_complete_message(self._progress_messages[name][1])


class SpikeDetekt(EventEmitter):
    """Spike detection class.

    Parameters
    ----------

    tempdir : str
        Path to the temporary directory used by the algorithm. It should be
        on a SSD for best performance.
    probe : dict
        The probe dictionary.
    **kwargs : dict
        Spike detection parameters.

    """
    def __init__(self, tempdir=None, probe=None, **kwargs):
        super(SpikeDetekt, self).__init__()
        self._tempdir = tempdir
        self._dead_channels = None
        # Load a probe.
        if probe is not None:
            kwargs['probe_channels'] = _channels_per_group(probe)
            kwargs['probe_adjacency_list'] = _probe_adjacency_list(probe)
        self._kwargs = kwargs
        self._n_channels_per_group = {
            group: len(channels)
            for group, channels in self._kwargs['probe_channels'].items()
        }
        self._groups = sorted(self._n_channels_per_group)
        self._n_features = self._kwargs['n_features_per_channel']
        before = self._kwargs['extract_s_before']
        after = self._kwargs['extract_s_after']
        self._n_samples_waveforms = before + after

    # Processing objects creation
    # -------------------------------------------------------------------------

    def _create_filter(self):
        rate = self._kwargs['sample_rate']
        low = self._kwargs['filter_low']
        high = self._kwargs['filter_high_factor'] * rate
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
        return WaveformExtractor(extract_before=before,
                                 extract_after=after,
                                 weight_power=weight_power,
                                 channels_per_group=probe_channels,
                                 thresholds=thresholds,
                                 )

    def _create_pca(self):
        n_pcs = self._kwargs['n_features_per_channel']
        return PCA(n_pcs=n_pcs)

    # Misc functions
    # -------------------------------------------------------------------------

    def update_params(self, **kwargs):
        self._kwargs.update(kwargs)

    # Processing functions
    # -------------------------------------------------------------------------

    def apply_filter(self, data):
        """Filter the traces."""
        filter = self._create_filter()
        return filter(data).astype(np.float32)

    def find_thresholds(self, traces):
        """Find weak and strong thresholds in filtered traces."""
        rate = self._kwargs['sample_rate']
        n_excerpts = self._kwargs['n_excerpts']
        excerpt_size = int(self._kwargs['excerpt_size_seconds'] * rate)
        single = bool(self._kwargs['use_single_threshold'])
        strong_f = self._kwargs['threshold_strong_std_factor']
        weak_f = self._kwargs['threshold_weak_std_factor']

        info("Finding the thresholds...")
        excerpt = get_excerpts(traces,
                               n_excerpts=n_excerpts,
                               excerpt_size=excerpt_size)
        excerpt_f = self.apply_filter(excerpt)
        thresholds = compute_threshold(excerpt_f,
                                       single_threshold=single,
                                       std_factor=(weak_f, strong_f))
        debug("Thresholds: {}.".format(thresholds))
        return {'weak': thresholds[0],
                'strong': thresholds[1]}

    def detect(self, traces_f, thresholds=None, dead_channels=None):
        """Detect connected waveform components in filtered traces.

        Parameters
        ----------

        traces_f : array
            An `(n_samples, n_channels)` array with the filtered data.
        thresholds : dict
            The weak and strong thresholds.
        dead_channels : array-like
            Array of dead channels.

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
        # Force crossings to be False on dead channels.
        if dead_channels is not None and len(dead_channels):
            assert dead_channels.max() < traces_f.shape[1]
            weak[:, dead_channels] = 0
            strong[:, dead_channels] = 0
        else:
            debug("No dead channels specified.")
        # Run the detection.
        detector = self._create_detector()
        return detector(weak_crossings=weak,
                        strong_crossings=strong)

    def extract_spikes(self, components, traces_f,
                       thresholds=None, keep_bounds=None):
        """Extract spikes from connected components.

        Returns a split object.

        Parameters
        ----------
        components : list
            List of connected components.
        traces_f : array
            Filtered data.
        thresholds : dict
            The weak and strong thresholds.
        keep_bounds : tuple
            (keep_start, keep_end).

        """
        n_spikes = len(components)
        if n_spikes == 0:
            return {}

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
        groups = np.array(groups, dtype=np.int32)
        assert groups.shape == (n_spikes,)
        assert groups.dtype == np.int32

        samples = np.array(samples, dtype=np.float64)
        assert samples.shape == (n_spikes,)
        assert samples.dtype == np.float64

        # These are lists of arrays of various shapes (because of various
        # groups).
        waveforms = _array_list(waveforms)
        assert waveforms.shape == (n_spikes,)
        assert waveforms.dtype == np.object

        masks = _array_list(masks)
        assert masks.dtype == np.object
        assert masks.shape == (n_spikes,)

        # Reorder the spikes.
        idx = np.argsort(samples)
        groups = groups[idx]
        samples = samples[idx]
        waveforms = waveforms[idx]
        masks = masks[idx]

        # Remove spikes in the overlapping bands.
        # WARNING: add keep_start to spike_samples, because spike_samples
        # is relative to the start of the chunk.
        (keep_start, keep_end) = keep_bounds
        idx = _keep_spikes(samples + keep_start, (keep_start, keep_end))

        # Split the data according to the channel groups.
        split = _split_spikes(groups, idx=idx, spike_samples=samples,
                              waveforms=waveforms, masks=masks)
        # split: {group: {'spike_samples': ..., 'waveforms':, 'masks':}}
        return split

    def waveform_pcs(self, waveforms, masks):
        """Compute waveform principal components.

        Returns
        -------

        pcs : array
            An `(n_features, n_samples, n_channels)` array.

        """
        pca = self._create_pca()
        if not waveforms.shape[0]:
            return
        assert (waveforms.shape[0], waveforms.shape[2]) == masks.shape
        return pca.fit(waveforms, masks)

    def features(self, waveforms, pcs):
        """Extract features from waveforms.

        Returns
        -------

        features : array
            An `(n_spikes, n_channels, n_features)` array.

        """
        pca = self._create_pca()
        out = pca.transform(waveforms, pcs=pcs)
        assert out.dtype == np.float32
        return out

    # Internal functions
    # -------------------------------------------------------------------------

    def _path(self, name, key=None, group=None):
        if self._tempdir is None:
            raise ValueError("The temporary directory must be specified.")
        assert key >= 0
        assert group is None or group >= 0
        path = '{group}/{name}/{chunk}.npy'.format(
            chunk=key, name=name, group=group if group is not None else 'all')
        path = op.realpath(op.join(self._tempdir, path))
        _ensure_dir_exists(op.dirname(path))
        return path

    def _save(self, array, name, key=None, group=None):
        path = self._path(name, key=key, group=group)
        if isinstance(array, list):
            _save_arrays(path, array)
        else:
            dtype = array.dtype
            assert dtype != np.object
            np.save(path, array)

    def _load(self, name, key=None, group=None, multiple_arrays=False):
        path = self._path(name, key=key, group=group)
        if not op.exists(path):
            return
        if multiple_arrays:
            return _load_arrays(path)
        else:
            return np.load(path)

    def _delete(self, name, key=None, group=None, multiple_arrays=False):
        path = self._path(name, key=key, group=group)
        if multiple_arrays:
            os.remove(path)
            os.remove(op.splitext(path)[0] + '.offsets.npy')
        else:
            os.remove(path)

    # Chunking
    # -------------------------------------------------------------------------

    def iter_chunks(self, n_samples):
        """Iterate over chunks."""
        rate = self._kwargs['sample_rate']
        chunk_size = int(self._kwargs['chunk_size_seconds'] * rate)
        overlap = int(self._kwargs['chunk_overlap_seconds'] * rate)
        for chunk_idx, bounds in enumerate(chunk_bounds(n_samples, chunk_size,
                                                        overlap=overlap)):
            yield Bunch(bounds=bounds,
                        s_start=bounds[0],
                        s_end=bounds[1],
                        keep_start=bounds[2],
                        keep_end=bounds[3],
                        keep_bounds=(bounds[2:4]),
                        key=bounds[2],
                        chunk_idx=chunk_idx,
                        )

    def n_chunks(self, n_samples):
        """Number of chunks."""
        return len(list(self.iter_chunks(n_samples)))

    # Output data
    # -------------------------------------------------------------------------

    def output_data(self,
                    n_samples,
                    n_channels,
                    groups=None,
                    spike_counts=None,
                    ):
        """Bunch of values to be returned by the algorithm."""
        n_samples_per_chunk = {chunk.key: (chunk.s_end - chunk.s_start)
                               for chunk in self.iter_chunks(n_samples)}
        keys = sorted(n_samples_per_chunk.keys())

        def _add_offset(chunk, group):
            samples = self._load('spike_samples', key=chunk.key, group=group)
            if samples is None:
                return samples
            return samples + chunk.s_start

        spike_samples = {group: (_add_offset(chunk, group)
                                 for chunk in self.iter_chunks(n_samples))
                         for group in groups}

        def _load(name):
            return {group: (self._load(name, key=chunk.key, group=group)
                            for chunk in self.iter_chunks(n_samples))
                    for group in groups}

        output = Bunch(n_chunks=len(keys),
                       groups=groups,
                       chunk_keys=keys,
                       spike_samples=spike_samples,
                       masks=_load('masks'),
                       features=_load('features'),
                       spike_counts=spike_counts,
                       n_spikes_total=spike_counts(),
                       n_spikes_per_group={group: spike_counts(group=group)
                                           for group in groups},
                       n_spikes_per_chunk={chunk: spike_counts(chunk=chunk)
                                           for chunk in keys},
                       )
        return output

    # Main loop
    # -------------------------------------------------------------------------

    def _iter_spikes(self, n_samples, step_spikes=1, thresholds=None):
        """Iterate over extracted spikes (possibly subset).

        Yield a split dictionary `{group: {'waveforms': ..., ...}}`.

        """
        for chunk in self.iter_chunks(n_samples):

            # Extract a few components.
            components = self._load('components', chunk.key,
                                    multiple_arrays=True)
            if components is None:
                continue

            k = np.clip(step_spikes, 1, len(components))
            components = components[::k]
            if not len(components):
                yield chunk, {}
                continue

            # Get the filtered chunk.
            chunk_f = self._load('filtered', key=chunk.key)

            # Extract the spikes from the chunk.
            split = self.extract_spikes(components, chunk_f,
                                        keep_bounds=chunk.keep_bounds,
                                        thresholds=thresholds)

            yield chunk, split

    def step_detect(self, pr=None, traces=None, thresholds=None):
        n_samples, n_channels = traces.shape
        n_chunks = self.n_chunks(n_samples)

        # Pass 1: find the connected components and count the spikes.
        pr.start_step('detect', n_chunks)

        # Dictionary {chunk_key: components}.
        # Every chunk has a unique key: the `keep_start` integer.
        n_spikes_total = 0
        for chunk in self.iter_chunks(n_samples):
            chunk_data = data_chunk(traces, chunk.bounds, with_overlap=True)

            # Apply the filter.
            data_f = self.apply_filter(chunk_data)
            assert data_f.dtype == np.float32
            assert data_f.shape == chunk_data.shape

            # Save the filtered chunk.
            self._save(data_f, 'filtered', key=chunk.key)

            # Detect spikes in the filtered chunk.
            components = self.detect(data_f, thresholds=thresholds,
                                     dead_channels=self._dead_channels)
            self._save(components, 'components', key=chunk.key)

            # Report progress.
            n_spikes_chunk = len(components)
            n_spikes_total += n_spikes_chunk
            pr.increment(n_spikes=n_spikes_chunk,
                         n_spikes_total=n_spikes_total)

        return n_spikes_total

    def step_excerpt(self, pr=None, n_samples=None,
                     n_spikes_total=None, thresholds=None):
        pr.start_step('excerpt', self.n_chunks(n_samples))

        k = int(n_spikes_total / float(self._kwargs['pca_n_waveforms_max']))
        w_subset = defaultdict(list)
        m_subset = defaultdict(list)
        n_spikes_total = 0
        for chunk, split in self._iter_spikes(n_samples, step_spikes=k,
                                              thresholds=thresholds):
            n_spikes_chunk = 0
            for group, out in split.items():
                w_subset[group].append(out['waveforms'])
                m_subset[group].append(out['masks'])
                assert len(out['masks']) == len(out['waveforms'])
                n_spikes_chunk += len(out['masks'])

            n_spikes_total += n_spikes_chunk
            pr.increment(n_spikes=n_spikes_chunk,
                         n_spikes_total=n_spikes_total)
        for group in self._groups:
            n_channels = self._n_channels_per_group[group]

            shape_w = (self._n_samples_waveforms, n_channels)
            w_subset[group] = _concatenate(w_subset[group], shape_w)

            shape_m = (n_channels,)
            m_subset[group] = _concatenate(m_subset[group], shape_m)

        return w_subset, m_subset

    def step_pcs(self, pr=None, w_subset=None, m_subset=None):
        pr.start_step('pca', len(self._groups))
        pcs = {}
        for group in self._groups:
            # Perform PCA and return the components.
            pcs[group] = self.waveform_pcs(w_subset[group],
                                           m_subset[group])
            pr.increment()
        return pcs

    def step_features(self, pr=None, n_samples=None,
                      pcs=None, thresholds=None):
        pr.start_step('extract', self.n_chunks(n_samples))
        chunk_counts = defaultdict(dict)  # {group: {key: n_spikes}}.
        n_spikes_total = 0
        for chunk, split in self._iter_spikes(n_samples,
                                              thresholds=thresholds):
            # split: {group: {'spike_samples': ..., 'waveforms':, 'masks':}}
            for group, out in split.items():
                out['features'] = self.features(out['waveforms'], pcs[group])
                n_spikes_chunk = len(out['spike_samples'])
                n_spikes_total += n_spikes_chunk
                chunk_counts[group][chunk.key] = n_spikes_chunk

                # Save the arrays.
                for name in ('spike_samples', 'features', 'masks'):
                    assert out[name].shape[0] == n_spikes_chunk
                    self._save(out[name], name, key=chunk.key, group=group)
            pr.increment(n_spikes_total=n_spikes_total)
        return chunk_counts

    def run_serial(self, traces, interval_samples=None):
        """Run SpikeDetekt using one CPU."""
        traces, offset = _cut_traces(traces, interval_samples)
        n_samples, n_channels = traces.shape

        # Find the weak and strong thresholds.
        thresholds = self.find_thresholds(traces)

        # Find dead channels.
        probe_channels = self._kwargs['probe_channels']
        self._dead_channels = _find_dead_channels(probe_channels, n_channels)

        # Create the progress reporter.
        n_chunks = self.n_chunks(n_samples)
        pr = SpikeDetektProgress(n_chunks=n_chunks)

        # Spike detection.
        n_spikes_total = self.step_detect(pr=pr, traces=traces,
                                          thresholds=thresholds)

        # Excerpt waveforms.
        w_subset, m_subset = self.step_excerpt(pr=pr,
                                               n_samples=n_samples,
                                               n_spikes_total=n_spikes_total,
                                               thresholds=thresholds)

        # Compute the PCs.
        pcs = self.step_pcs(pr=pr, w_subset=w_subset, m_subset=m_subset)

        # Compute all features.
        chunk_counts = self.step_features(pr=pr, n_samples=n_samples,
                                          pcs=pcs, thresholds=thresholds)

        spike_counts = SpikeCounts(chunk_counts)
        return self.output_data(n_samples, n_channels,
                                self._groups, spike_counts)
