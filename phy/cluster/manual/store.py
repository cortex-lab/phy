# -*- coding: utf-8 -*-

"""Cluster store."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os
import os.path as op
import re

import numpy as np

from ._utils import (_concatenate_per_cluster_arrays,
                     _subset_spikes_per_cluster,
                     )
from ...utils.logging import debug, info
from ...ext.six import string_types, integer_types


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


#------------------------------------------------------------------------------
# Memory store
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


#------------------------------------------------------------------------------
# Disk store
#------------------------------------------------------------------------------

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
        # Example of filename: `123.mykey`.
        cluster = _as_int(cluster)
        filename = '{0:d}.{1:s}'.format(cluster, key)
        return op.realpath(op.join(self._directory, filename))

    def _cluster_file_exists(self, cluster, key):
        """Return whether a cluster file exists."""
        cluster = _as_int(cluster)
        return op.exists(self._cluster_path(cluster, key))

    def _is_cluster_file(self, path):
        """Return whether a filename is of the form `xxx.yyy` where xxx is a
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
    """Hold per-cluster information on disk and in memory.

    Note
    ----

    Currently, this is used to accelerate access to features, masks, and
    cluster statistics. Features and masks of all clusters are stored in a
    disk cache. Cluster statistics are computed when loading a dataset,
    and are kept in memory afterwards. All data is dynamically updated
    when clustering changes occur.

    """
    def __init__(self,
                 model=None,
                 spike_clusters=None,
                 spikes_per_cluster=None,
                 path=None,
                 ):
        self._model = model
        self._spike_clusters = spike_clusters
        self._spikes_per_cluster = spikes_per_cluster
        self._memory = MemoryStore()
        self._disk = DiskStore(path) if path is not None else None
        self._items = []
        self._locations = {}

    def _store(self, location):
        if location == 'memory':
            return self._memory
        elif location == 'disk':
            return self._disk
        else:
            raise ValueError("The `location` should be `memory` "
                             "or `disk`.")

    @property
    def memory_store(self):
        """Hold some cluster statistics."""
        return self._memory

    @property
    def disk_store(self):
        """Manage the cache of per-cluster voluminous data."""
        return self._disk

    @property
    def spikes_per_cluster(self):
        """Dictionary `{cluster_id: spike_ids}`."""
        return self._spikes_per_cluster

    # @spikes_per_cluster.setter
    # def spikes_per_cluster(self, value):
    #     """Update the `spikes_per_cluster` structure."""
    #     assert isinstance(value, dict)
    #     self._spikes_per_cluster = value

    @property
    def spike_clusters(self):
        """Spike clusters array."""
        return self._spike_clusters

    # @spike_clusters.setter
    # def spike_clusters(self, value):
    #     """Update the `spike_clusters` array."""
    #     assert isinstance(value, np.ndarray)
    #     self._spike_clusters = value

    def update_spikes_per_cluster(self, spikes_per_cluster):
        self._spikes_per_cluster = spikes_per_cluster
        for item in self._items:
            item.spikes_per_cluster = spikes_per_cluster

    @property
    def cluster_ids(self):
        """All cluster ids appearing in the `spikes_per_cluster` dictionary."""
        return sorted(self._spikes_per_cluster)

    @property
    def store_items(self):
        """List of registered store items."""
        return self._items

    def register_field(self, name, location, dtype=None, shape=None):
        """Register a new piece of data to store on memory or on disk.

        Parameters
        ----------

        name : str
            The name of the field.
        location : str
            `memory` or `disk`.
        dtype : NumPy dtype or None
            The dtype of arrays stored for that field. This is only used when
            the location is `disk`.
        shape : tuple or None
            The shape of arrays. This is only used when the location is `disk`.
            This is used by `np.reshape()`, so the shape can contain a `-1`.

        Notes
        -----

        When storing information to disk, only NumPy arrays are supported
        currently. They are saved as flat binary files. This is why the
        dtype and shape must be registered here, otherwise that information
        is lost. This metadata is not saved in the files.

        """
        # HACK: need to use a factory function because in Python
        # functions are closed over names, not values. Here we
        # want `name` to refer to the `name` local variable.
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

    def register_item(self, item_cls, **kwargs):
        """Register a `StoreItem` class in the store.

        A `StoreItem` class is responsible for storing some data to disk
        and memory. It must register one or several pieces of data.

        """

        # Instantiate the item.
        item = item_cls(model=self._model,
                        memory_store=self._memory,
                        disk_store=self._disk,
                        spikes_per_cluster=self.spikes_per_cluster,
                        spike_clusters=self.spike_clusters,
                        **kwargs)
        assert item.fields is not None

        # Register all fields declared by the store item.
        for field in item.fields:

            name, location = field[:2]
            dtype = field[2] if len(field) >= 3 else None
            shape = field[3] if len(field) == 4 else None

            self.register_field(name, location, dtype=dtype, shape=shape)

        # Register the StoreItem instance.
        self._items.append(item)

    def load(self, name, clusters, spikes):
        """Load some data for a number of clusters and spikes."""
        # Ensure clusters and spikes are sorted and do not have duplicates.
        clusters = np.unique(clusters)
        spikes = np.unique(spikes)
        load = getattr(self, name)
        # Get spikes_per_cluster and data arrays for the specified spikes.
        spc = {cluster: self._spikes_per_cluster[cluster]
               for cluster in clusters}
        arrays = {cluster: load(cluster) for cluster in clusters}
        spc_s, arrays_s = _subset_spikes_per_cluster(spc, arrays, spikes)
        # Return the concatenated array.
        return _concatenate_per_cluster_arrays(spc_s, arrays_s)

    def on_cluster(self, up):
        """Update the cluster store when clustering changes occur.

        This method calls `item.on_cluster(up)` on all registered store items.

        """
        # No need to delete the old clusters from the store, we can keep
        # them for possible undo, and regularly clean up the store.
        for item in self._items:
            item.on_cluster(up)

    # Files
    #--------------------------------------------------------------------------

    @property
    def path(self):
        """Path to the disk store cache."""
        return self.disk_store.path

    @property
    def old_clusters(self):
        """Clusters in the disk store that are no longer in the clustering."""
        return sorted(set(self.disk_store.cluster_ids) -
                      set(self.cluster_ids))

    @property
    def files(self):
        """List of files present in the disk store."""
        return self.disk_store.files

    # Status
    #--------------------------------------------------------------------------

    @property
    def total_size(self):
        """Total size of the disk store."""
        return _directory_size(self.path)

    def is_consistent(self):
        """Return whether the cluster store is probably consistent.

        Return true if all cluster stores files exist and have the expected
        file size.

        """
        valid = set(self.cluster_ids)
        # All store items should be consistent on all valid clusters.
        consistent = all(all(item.is_consistent(clu,
                             self.spikes_per_cluster.get(clu, []))
                             for clu in valid)
                         for item in self._items)
        return consistent

    @property
    def status(self):
        """Return the current status of the cluster store."""
        in_store = set(self.disk_store.cluster_ids)
        valid = set(self.cluster_ids)
        invalid = in_store - valid

        n_store = len(in_store)
        n_old = len(invalid)
        size = self.total_size / (1024. ** 2)
        consistent = str(self.is_consistent()).rjust(5)

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
        """Display the current status of the cluster store."""
        print(self.status)

    # Store management
    #--------------------------------------------------------------------------

    def clear(self):
        """Erase all files in the store."""
        self.memory_store.clear()
        self.disk_store.clear()
        info("Cluster store cleared.")

    def clean(self):
        """Erase all old files in the store."""
        to_delete = self.old_clusters
        self.memory_store.erase(to_delete)
        self.disk_store.erase(to_delete)
        n = len(to_delete)
        info("{0} clusters deleted from the cluster store.".format(n))

    def generate(self,
                 mode=None,
                 ):
        """Generate the cluster store.

        Parameters
        ----------

        mode : str (default is None)
            How the cluster store should be generated. Options are:

            * None or `default`: only regenerate the missing or inconsistent
              clusters
            * `force`: fully regenerate the cluster
            * `read-only`: just load the existing files, do not write anything

        """
        assert isinstance(self._spikes_per_cluster, dict)
        # assert isinstance(self._spike_clusters, np.ndarray)
        if hasattr(self._model, 'name'):
            name = self._model.name
        else:
            name = 'the current model'
        debug("Initializing the cluster store for {0:s}...".format(name))
        for item in self._items:
            item.store_all_clusters(mode)
        debug("Done!")


class StoreItem(object):
    """A class describing information stored in the cluster store.

    Parameters
    ----------

    fields : list
        A list of pairs `(field_name, storage_location)`.
        `storage_location` is either `memory` or `disk`.
    model : Model
        A `Model` instance for the current dataset.
    memory_store : MemoryStore
        The `MemoryStore` instance for the current dataset.
    disk_store : DiskStore
        The DiskStore instance for the current dataset.

    """
    fields = None  # list of `(field_name, storage_location)`
    name = 'item'

    def __init__(self,
                 model=None,
                 memory_store=None,
                 disk_store=None,
                 spike_clusters=None,
                 spikes_per_cluster=None,
                 ):
        self.model = model
        self.memory_store = memory_store
        self.disk_store = disk_store
        self._spikes_per_cluster = spikes_per_cluster
        self._spike_clusters = spike_clusters

    @property
    def spikes_per_cluster(self):
        """Spikes per cluster."""
        return self._spikes_per_cluster

    @spikes_per_cluster.setter
    def spikes_per_cluster(self, value):
        self._spikes_per_cluster = value

    @property
    def spike_clusters(self):
        """Spikes per cluster."""
        return self._spike_clusters

    # @spike_clusters.setter
    # def spike_clusters(self, value):
    #     self._spike_clusters = value

    @property
    def cluster_ids(self):
        """Array of cluster ids."""
        return sorted(self._spikes_per_cluster)

    def is_consistent(self, cluster, spikes):
        """Return whether the stored item is consistent.

        To be overriden."""
        return False

    def to_generate(self, mode=None):
        """Return the list of clusters that need to be regenerated."""
        if mode in (None, 'default'):
            return [cluster for cluster in self.cluster_ids
                    if not self.is_consistent(cluster,
                                              self.spikes_per_cluster[cluster],
                                              )]
        elif mode == 'force':
            return self.cluster_ids
        elif mode == 'read-only':
            return []
        else:
            raise ValueError("`mode` should be None, `default`, `force`, "
                             "or `read-only`.")

    def store_cluster(self, cluster, spikes=None, mode=None):
        """Store data for a cluster from the model to the store.

        May be overridden.

        No need to delete old clusters here.

        """
        pass

    def store_all_clusters(self, mode=None):
        """Copy all data for that item from the model to the cluster store."""
        for cluster in self.to_generate(mode):
            debug("Loading {0:s}, cluster {1:d}...".format(self.name,
                  cluster))
            self.store_cluster(cluster,
                               spikes=self._spikes_per_cluster[cluster],
                               mode=mode,
                               )

    def on_cluster(self, up):
        """Update the stored data when a clustering change happens.

        May be overridden.

        No need to delete old clusters here."""
        for cluster in up.added:
            self.store_cluster(cluster, up.new_spikes_per_cluster[cluster])
