# -*- coding: utf-8 -*-

"""Cluster store."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os
import os.path as op
import re

import numpy as np

from ...utils.array import _is_array_like, _index_of
from ...utils.logging import debug
from ...ext.six import string_types, integer_types
from ...utils.event import ProgressReporter


#------------------------------------------------------------------------------
# Utility functions
#------------------------------------------------------------------------------

def _directory_size(path):
    """Return the total size in bytes of a directory."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


#------------------------------------------------------------------------------
# Data stores
#------------------------------------------------------------------------------

class MemoryStore(object):
    """Store cluster-related data in memory."""
    def __init__(self):
        self._ds = {}

    def store(self, cluster, **data):
        """Store cluster-related data."""
        if cluster not in self._ds:
            self._ds[cluster] = {}
        self._ds[cluster].update(data)

    def load(self, cluster, keys=None):
        """Load cluster-related data."""
        if keys is None:
            return self._ds.get(cluster, {})
        else:
            if isinstance(keys, string_types):
                return self._ds.get(cluster, {}).get(keys, None)
            assert isinstance(keys, (list, tuple))
            return {key: self._ds.get(cluster, {}).get(key, None)
                    for key in keys}

    @property
    def cluster_ids(self):
        """List of cluster ids in the store."""
        return sorted(self._ds.keys())

    def erase(self, clusters):
        """Delete some clusters from the store."""
        assert isinstance(clusters, list)
        for cluster in clusters:
            if cluster in self._ds:
                del self._ds[cluster]

    def clear(self):
        """Clear the store completely by deleting all clusters."""
        self.erase(self.cluster_ids)


def _load_ndarray(f, dtype=None, shape=None):
    if dtype is None:
        return f
    else:
        arr = np.fromfile(f, dtype=dtype)
        if shape is not None:
            arr = arr.reshape(shape)
        return arr


def _as_int(x):
    if isinstance(x, integer_types):
        return x
    x = np.asscalar(x)
    return x


def _file_cluster_id(path):
    return int(op.splitext(op.basename(path))[0])


class DiskStore(object):
    """Store cluster-related data in HDF5 files."""
    def __init__(self, directory):
        assert directory is not None
        # White list of extensions, to be sure we don't erase
        # the wrong files.
        self._allowed_extensions = set()
        self._directory = op.realpath(op.expanduser(directory))

    @property
    def path(self):
        return self._directory

    # Internal methods
    # -------------------------------------------------------------------------

    def _check_extension(self, file):
        """Check that a file extension belongs to the white list of
        allowed extensions. This is for safety."""
        _, extension = op.splitext(file)
        extension = extension[1:]
        if extension not in self._allowed_extensions:
            raise RuntimeError("The extension '{0}' ".format(extension) +
                               "hasn't been registered.")

    def _cluster_path(self, cluster, key):
        """Return the absolute path of a cluster in the disk store."""
        # TODO: subfolders
        # Example of filename: '123.mykey'.
        cluster = _as_int(cluster)
        filename = '{0:d}.{1:s}'.format(cluster, key)
        return op.realpath(op.join(self._directory, filename))

    def _cluster_file_exists(self, cluster, key):
        """Return whether a cluster file exists."""
        cluster = _as_int(cluster)
        return op.exists(self._cluster_path(cluster, key))

    def _is_cluster_file(self, path):
        """Return whether a filename is of the form 'xxx.yyy' where xxx is a
        numbe and yyy belongs to the set of allowed extensions."""
        filename = op.basename(path)
        extensions = '({0})'.format('|'.join(sorted(self._allowed_extensions)))
        regex = r'^[0-9]+\.' + extensions + '$'
        return re.match(regex, filename) is not None

    # Public methods
    # -------------------------------------------------------------------------

    def register_file_extensions(self, extensions):
        """Register file extensions explicitely. This is a security
        to make sure that we don't accidentally delete the wrong files."""
        if isinstance(extensions, string_types):
            extensions = [extensions]
        assert isinstance(extensions, list)
        for extension in extensions:
            self._allowed_extensions.add(extension)

    def store(self, cluster, append=False, **data):
        """Store a NumPy array to disk."""
        # Do not create the file if there's nothing to write.
        if not data:
            return
        mode = 'wb' if not append else 'ab'
        for key, value in data.items():
            assert isinstance(value, np.ndarray)
            path = self._cluster_path(cluster, key)
            self._check_extension(path)
            assert self._is_cluster_file(path)
            with open(path, mode) as f:
                value.tofile(f)

    def _get(self, cluster, key, dtype=None, shape=None):
        # The cluster doesn't exist: return None for all keys.
        if not self._cluster_file_exists(cluster, key):
            return None
        else:
            with open(self._cluster_path(cluster, key), 'rb') as f:
                return _load_ndarray(f, dtype=dtype, shape=shape)

    def load(self, cluster, keys, dtype=None, shape=None):
        """Load cluster-related data. Return a file handle, to be used
        with np.fromfile() once the dtype and shape are known."""
        assert keys is not None
        if isinstance(keys, string_types):
            return self._get(cluster, keys, dtype=dtype, shape=shape)
        assert isinstance(keys, list)
        out = {}
        for key in keys:
            out[key] = self._get(cluster, key, dtype=dtype, shape=shape)
        return out

    @property
    def files(self):
        """List of files present in the directory."""
        if not op.exists(self._directory):
            return []
        return sorted(filter(self._is_cluster_file,
                             os.listdir(self._directory)))

    @property
    def cluster_ids(self):
        """List of cluster ids in the store."""
        clusters = set([_file_cluster_id(file) for file in self.files])
        return sorted(clusters)

    def erase(self, clusters):
        """Delete some clusters from the store."""
        for cluster in clusters:
            for key in self._allowed_extensions:
                path = self._cluster_path(cluster, key)
                if not op.exists(path):
                    continue
                # Safety first: http://bit.ly/1ITJyF6
                self._check_extension(path)
                if self._is_cluster_file(path):
                    os.remove(path)
                else:
                    raise RuntimeError("The file {0} was about ".format(path) +
                                       "to be removed, but it doesn't appear "
                                       "to be a valid cluster file.")

    def clear(self):
        """Clear the store completely by deleting all clusters."""
        self.erase(self.cluster_ids)


#------------------------------------------------------------------------------
# Cluster store
#------------------------------------------------------------------------------

class ClusterStore(object):
    def __init__(self, model=None, path=None):
        self._model = model
        self._spikes_per_cluster = {}
        self._memory = MemoryStore()
        self._disk = DiskStore(path) if path is not None else None
        self._items = []
        self._locations = {}
        self._progress_reporter = ProgressReporter()

    def _store(self, location):
        if location == 'memory':
            return self._memory
        elif location == 'disk':
            return self._disk
        else:
            raise ValueError("The 'location' should be 'memory' "
                             "or 'disk'.")

    @property
    def memory_store(self):
        return self._memory

    @property
    def disk_store(self):
        return self._disk

    @property
    def progress_reporter(self):
        return self._progress_reporter

    @property
    def spikes_per_cluster(self):
        return self._spikes_per_cluster

    def register_item(self, item_cls):
        """Register a StoreItem instance in the store."""

        # Instantiate the item.
        item = item_cls(model=self._model,
                        memory_store=self._memory,
                        disk_store=self._disk,
                        progress_reporter=self._progress_reporter,
                        )
        assert item.fields is not None

        for field in item.fields:
            name, location = field[:2]
            dtype = field[2] if len(field) >= 3 else None
            shape = field[3] if len(field) == 4 else None

            # HACK: need to use a factory function because in Python
            # functions are closed over names, not values. Here we
            # want 'name' to refer to the 'name' local variable.
            def _make_func(name, location):
                kwargs = {} if location == 'memory' else {'dtype': dtype,
                                                          'shape': shape}
                return lambda cluster: self._store(location).load(cluster,
                                                                  name,
                                                                  **kwargs)

            # Register the item location (memory or store).
            assert name not in self._locations
            if self._disk:
                self._disk.register_file_extensions(name)
            self._locations[name] = location

            # Get the load function.
            load = _make_func(name, location)

            # We create the self.<name>(cluster) method for loading.
            # We need to ensure that the method name isn't already attributed.
            assert not hasattr(self, name)
            setattr(self, name, load)

        # Register the StoreItem instance.
        self._items.append(item)

    def load(self, name, clusters, spikes):
        assert _is_array_like(clusters)
        load = getattr(self, name)

        # Concatenation of arrays for all clusters.
        arrays = np.concatenate([load(cluster) for cluster in clusters])
        # Concatenation of spike indices for all clusters.
        spike_clusters = np.concatenate([self._spikes_per_cluster[cluster]
                                         for cluster in clusters])
        assert np.all(np.in1d(spikes, spike_clusters))
        idx = _index_of(spikes, spike_clusters)
        return arrays[idx, ...]

    def on_cluster(self, up):
        # No need to delete the old clusters from the store, we can keep
        # them for possible undo, and regularly clean up the store.

        # self._memory.erase(up.deleted)
        # self._disk.erase(up.deleted)

        for item in self._items:
            item.on_cluster(up)

    # Store management
    #--------------------------------------------------------------------------

    @property
    def path(self):
        return self.disk_store.path

    @property
    def old_clusters(self):
        return sorted(set(self.disk_store.cluster_ids) -
                      set(self.model.cluster_ids))

    @property
    def files(self):
        return [op.join(self.path, file) for file in self.disk_store.files]

    @property
    def total_size(self):
        return _directory_size(self.path)

    @property
    def status(self):
        in_store = set(self.disk_store.cluster_ids)
        valid = set(self._model.cluster_ids)
        invalid = in_store - valid

        n_store = len(in_store)
        n_old = len(invalid)
        size = self.total_size / (1024. ** 2)
        # All store items should be consistent on all valid clusters.
        consistent = all(all(item.is_consistent(cluster)
                             for cluster in valid)
                         for item in self._items)
        consistent = str(consistent).rjust(5)

        status = ''
        header = "Cluster store status ({0})".format(self.path)
        status += header + '\n'
        status += '-' * len(header) + '\n'
        status += "Number of clusters in the store   {0: 4d}\n".format(n_store)
        status += "Number of old clusters            {0: 4d}\n".format(n_old)
        status += "Total size (MB)                {0: 7.0f}\n".format(size)
        status += "Consistent                       {0}\n".format(consistent)
        return status

    def display_status(self):
        """Display the current status of the store."""
        print(self.status)

    def clear(self):
        """Erase all files in the store."""
        self.memory_store.clear()
        self.disk_store.clear()

    def clean(self):
        """Erase all old files in the store."""
        to_delete = self.old_clusters
        self.memory_store.erase(to_delete)
        self.disk_store.erase(to_delete)

    def generate(self, spikes_per_cluster):
        """Generate the cluster store."""
        assert isinstance(spikes_per_cluster, dict)
        self._spikes_per_cluster = spikes_per_cluster
        if hasattr(self._model, 'name'):
            name = self._model.name
        else:
            name = 'the current model'
        debug("Initializing the cluster store for {0:s}...".format(name))
        for item in self._items:
            item.spikes_per_cluster = spikes_per_cluster
            item.store_all_clusters()
        debug("Done!")


class StoreItem(object):
    """A class describing information stored in the cluster store.

    Attributes
    ----------
    fields : list
        A list of pairs (field_name, storage_location).
        storage_location is either 'memory', 'disk'.
    model : Model
        A Model instance for the current dataset.
    memory_store : MemoryStore
        The MemoryStore instance for the current dataset.
    disk_store : DiskStore
        The DiskStore instance for the current dataset.
    progress_reporter : ProgressReporter
        The ProgressReporter instance for the current dataset.

    Methods
    -------
    store_cluster(cluster, spikes)
        Extract some data from the model and store it in the cluster store.
    on_cluster(up)
        Update the store when the clustering changes.

    """
    fields = None  # list of (field_name, storage_location)
    name = 'item'

    def __init__(self,
                 model=None,
                 memory_store=None,
                 disk_store=None,
                 progress_reporter=None,
                 ):
        self._spikes_per_cluster = None
        self.model = model
        self.memory_store = memory_store
        self.disk_store = disk_store
        self.progress_reporter = progress_reporter

    @property
    def spikes_per_cluster(self):
        return self._spikes_per_cluster

    @spikes_per_cluster.setter
    def spikes_per_cluster(self, value):
        self._spikes_per_cluster = value

    def store_all_clusters(self):
        """Copy all data for that item from the model to the cluster store."""
        clusters = sorted(self._spikes_per_cluster.keys())
        for cluster in clusters:
            debug("Loading {0:s}, cluster {1:d}...".format(self.name,
                  cluster))
            self.store_cluster(cluster, self._spikes_per_cluster[cluster])

    def is_consistent(self, cluster):
        """To be overriden."""
        return None

    def on_cluster(self, up):
        """May be overridden. No need to delete old clusters here."""
        for cluster in up.added:
            self.store_cluster(cluster, up.new_spikes_per_cluster[cluster])

    def store_cluster(self, cluster, spikes):
        """May be overridden. No need to delete old clusters here."""
        pass
