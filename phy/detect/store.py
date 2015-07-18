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

from ..utils._types import Bunch
from ..utils.array import (get_excerpts,
                           chunk_bounds,
                           data_chunk,
                           _as_array,
                           _save_arrays,
                           _load_arrays,
                           _concatenate,
                           )
from ..utils.logging import debug, info
from ..utils.settings import _ensure_dir_exists


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
        debug("Store {}.".format(path))

    def load(self, **kwargs):
        path = self._path(**kwargs)
        if not op.exists(path):
            debug("File `{}` doesn't exist.".format(path))
            return
        # Multiple arrays:
        debug("Load {}.".format(path))
        if self._contains_multiple_arrays(path):
            return _load_arrays(path)
        else:
            return np.load(path)

    def delete(self, **kwargs):
        path = self._path(**kwargs)
        # os.remove(path)
        print("remove", path)
        offsets_path = self._offsets_path(path)
        if op.exists(offsets_path):
            # os.remove(offsets_path)
            print("remove", offsets_path)
            debug("Deleted `{}`.".format(offsets_path))
        debug("Deleted `{}`.".format(path))


class SpikeDetektStore(ArrayStore):
    """Store the following items:

    * filtered
    * components
    * spike_samples
    * features
    * masks

    """

    def _path(self, name=None, key=None, group=None):
        assert key >= 0
        assert group is None or group >= 0
        assert isinstance(name, string_types)
        return 'group_{group}/{name}/chunk_{chunk:12d}.npy'.format(
            chunk=key, name=name, group=group if group is not None else 'all')

    @property
    def groups(self):
        pass

    @property
    def chunk_keys(self):
        pass

    def spike_samples(self, group):
        pass

    def features(self, group):
        """Yield chunk features."""

    def masks(self, group):
        """Yield chunk masks."""

    def spike_counts(self, group=None, chunk_key=None):
        pass

    def concatenate(self, arrays):
        pass
