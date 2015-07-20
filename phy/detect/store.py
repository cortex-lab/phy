# -*- coding: utf-8 -*-

"""Spike detection store."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os
import os.path as op
from collections import defaultdict

import numpy as np
from six import string_types

from ..utils.array import (_save_arrays,
                           _load_arrays,
                           _concatenate,
                           )
from ..utils.logging import debug
from ..utils.settings import _ensure_dir_exists


#------------------------------------------------------------------------------
# Spike counts
#------------------------------------------------------------------------------

class SpikeCounts(object):
    """Count spikes in chunks and channel groups."""
    def __init__(self, counts=None, groups=None, chunk_keys=None):
        self._groups = groups
        self._chunk_keys = chunk_keys
        self._counts = counts or defaultdict(lambda: defaultdict(int))

    def append(self, group=None, chunk_key=None, count=None):
        self._counts[group][chunk_key] += count

    @property
    def counts(self):
        return self._counts

    def per_group(self, group):
        return sum(self._counts.get(group, {}).values())

    def per_chunk(self, chunk_key):
        return sum(self._counts[group].get(chunk_key, 0)
                   for group in self._groups)

    def __call__(self, group=None, chunk_key=None):
        if group is not None and chunk_key is not None:
            return self._counts.get(group, {}).get(chunk_key, 0)
        elif group is not None:
            return self.per_group(group)
        elif chunk_key is not None:
            return self.per_chunk(chunk_key)
        elif group is None and chunk_key is None:
            return sum(self.per_group(group) for group in self._groups)


#------------------------------------------------------------------------------
# Spike detection store
#------------------------------------------------------------------------------

class ArrayStore(object):
    def __init__(self, root_dir):
        self._root_dir = op.realpath(root_dir)
        _ensure_dir_exists(self._root_dir)

    def _rel_path(self, **kwargs):
        """Relative to the root."""
        raise NotImplementedError()

    def _path(self, **kwargs):
        """Absolute path of a data file."""
        path = op.realpath(op.join(self._root_dir, self._rel_path(**kwargs)))
        _ensure_dir_exists(op.dirname(path))
        assert path.endswith('.npy')
        return path

    def _offsets_path(self, path):
        assert path.endswith('.npy')
        return op.splitext(path)[0] + '.offsets.npy'

    def _contains_multiple_arrays(self, path):
        return op.exists(path) and op.exists(self._offsets_path(path))

    def store(self, data=None, **kwargs):
        """Store an array or list of arrays."""
        path = self._path(**kwargs)
        if isinstance(data, list):
            if not data:
                return
            _save_arrays(path, data)
        elif isinstance(data, np.ndarray):
            dtype = data.dtype
            if not data.size:
                return
            assert dtype != np.object
            np.save(path, data)
        # debug("Store {}.".format(path))

    def load(self, **kwargs):
        path = self._path(**kwargs)
        if not op.exists(path):
            debug("File `{}` doesn't exist.".format(path))
            return
        # Multiple arrays:
        # debug("Load {}.".format(path))
        if self._contains_multiple_arrays(path):
            return _load_arrays(path)
        else:
            return np.load(path)

    def delete(self, **kwargs):
        path = self._path(**kwargs)
        if op.exists(path):
            os.remove(path)
            # debug("Deleted `{}`.".format(path))
        offsets_path = self._offsets_path(path)
        if op.exists(offsets_path):
            os.remove(offsets_path)
            # debug("Deleted `{}`.".format(offsets_path))


class SpikeDetektStore(ArrayStore):
    """Store the following items:

    * filtered
    * components
    * spike_samples
    * features
    * masks

    """
    def __init__(self, root_dir, groups=None, chunk_keys=None):
        super(SpikeDetektStore, self).__init__(root_dir)
        self._groups = groups
        self._chunk_keys = chunk_keys
        self._spike_counts = SpikeCounts(groups=groups, chunk_keys=chunk_keys)

    def _rel_path(self, name=None, chunk_key=None, group=None):
        assert chunk_key >= 0
        assert group is None or group >= 0
        assert isinstance(name, string_types)
        group = group if group is not None else 'all'
        return 'group_{group}/{name}/chunk_{chunk:d}.npy'.format(
            chunk=chunk_key, name=name, group=group)

    @property
    def groups(self):
        return self._groups

    @property
    def chunk_keys(self):
        return self._chunk_keys

    def _iter(self, group=None, name=None):
        for chunk_key in self.chunk_keys:
            yield self.load(group=group, chunk_key=chunk_key, name=name)

    def spike_samples(self, group=None):
        if group is None:
            return {group: self.spike_samples(group) for group in self._groups}
        return self.concatenate(self._iter(group=group, name='spike_samples'))

    def features(self, group=None):
        """Yield chunk features."""
        if group is None:
            return {group: self.features(group) for group in self._groups}
        return self._iter(group=group, name='features')

    def masks(self, group=None):
        """Yield chunk masks."""
        if group is None:
            return {group: self.masks(group) for group in self._groups}
        return self._iter(group=group, name='masks')

    @property
    def spike_counts(self):
        return self._spike_counts

    def append(self, group=None, chunk_key=None,
               spike_samples=None, features=None, masks=None,
               spike_offset=0):
        if spike_samples is None or len(spike_samples) == 0:
            return
        n = len(spike_samples)
        assert features.shape[0] == n
        assert masks.shape[0] == n
        spike_samples = spike_samples + spike_offset

        self.store(group=group, chunk_key=chunk_key,
                   name='features', data=features)
        self.store(group=group, chunk_key=chunk_key,
                   name='masks', data=masks)
        self.store(group=group, chunk_key=chunk_key,
                   name='spike_samples', data=spike_samples)
        self._spike_counts.append(group=group, chunk_key=chunk_key, count=n)

    def concatenate(self, arrays):
        return _concatenate(arrays)

    def delete_all(self, name):
        """Delete all files for a given data name."""
        for group in self._groups:
            for chunk_key in self._chunk_keys:
                super(SpikeDetektStore, self).delete(name=name, group=group,
                                                     chunk_key=chunk_key)
