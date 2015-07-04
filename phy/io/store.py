# -*- coding: utf-8 -*-

"""Cluster store."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from collections import OrderedDict
import os
import os.path as op
import re

import numpy as np

from ..utils._types import _as_int, _is_integer, _is_array_like
from ..utils._misc import _load_json, _save_json
from ..utils.array import (PerClusterData, _spikes_in_clusters,
                           _subset_spc, _load_ndarray)
from ..utils.event import ProgressReporter
from ..utils.logging import debug, info, warn
from ..ext.six import string_types


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


def _file_cluster_id(path):
    return int(op.splitext(op.basename(path))[0])


def _default_array(shape, value=0, n_spikes=0, dtype=np.float32):
    shape = (n_spikes,) + shape[1:]
    out = np.empty(shape, dtype=dtype)
    out.fill(value)
    return out


def _assert_per_cluster_data_compatible(d_0, d_1):
    n_0 = {k: len(v) for (k, v) in d_0.items()}
    n_1 = {k: len(v) for (k, v) in d_1.items()}
    if n_0 != n_1:
        raise IOError("Inconsistency in the cluster store: please remove "
                      "`./<basename>.phy/cluster_store/`.")


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

    def __contains__(self, item):
        return item in self._ds


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
            return _load_ndarray(self._cluster_path(cluster, key),
                                 dtype=dtype, shape=shape, lazy=False)

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

    def save_file(self, filename, data):
        path = op.realpath(op.join(self._directory, filename))
        _save_json(path, data)

    def load_file(self, filename):
        path = op.realpath(op.join(self._directory, filename))
        if not op.exists(path):
            return None
        try:
            return _load_json(path)
        except ValueError as e:
            warn("Error when loading `{}`: {}.".format(path, e))
            return None

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
# Store item
#------------------------------------------------------------------------------

class StoreItem(object):
    """A class describing information stored in the cluster store.

    Parameters
    ----------

    name : str
        Name of the item.
    fields : list
        A list of field names.
    model : Model
        A `Model` instance for the current dataset.
    memory_store : MemoryStore
        The `MemoryStore` instance for the current dataset.
    disk_store : DiskStore
        The DiskStore instance for the current dataset.

    """
    name = 'item'
    fields = None  # list of names

    def __init__(self, cluster_store=None):
        self.cluster_store = cluster_store
        self.model = cluster_store.model
        self.memory_store = cluster_store.memory_store
        self.disk_store = cluster_store.disk_store
        self._spikes_per_cluster = cluster_store.spikes_per_cluster
        self._pr = ProgressReporter()
        self._pr.set_progress_message('Initializing ' + self.name +
                                      ': {progress:.1f}%.')
        self._pr.set_complete_message(self.name.capitalize() + ' initialized.')
        self._shapes = {}

    def empty_values(self, name):
        """Return a null array of the right shape for a given field."""
        return _default_array(self._shapes.get(name, (-1,)), value=0.)

    @property
    def progress_reporter(self):
        """Progress reporter instance."""
        return self._pr

    @property
    def spikes_per_cluster(self):
        """Spikes per cluster."""
        return self._spikes_per_cluster

    @spikes_per_cluster.setter
    def spikes_per_cluster(self, value):
        self._spikes_per_cluster = value

    def spikes_in_clusters(self, clusters):
        """Return the spikes belonging to clusters."""
        return _spikes_in_clusters(self._spikes_per_cluster, clusters)

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

    def store(self, cluster):
        """Store data for a cluster from the model to the store.

        May be overridden.

        """
        pass

    def store_all(self, mode=None, **kwargs):
        """Copy all data for that item from the model to the cluster store."""
        clusters = self.to_generate(mode)
        self._pr.value_max = len(clusters)
        for cluster in clusters:
            self.store(cluster, **kwargs)
            self._pr.value += 1
        self._pr.set_complete()

    def load(self, cluster, name):
        """Load data for one cluster."""
        raise NotImplementedError()

    def load_multi(self, clusters, name):
        """Load data for several clusters."""
        raise NotImplementedError()

    def load_spikes(self, spikes, name):
        """Load data from an array of spikes."""
        raise NotImplementedError()

    def on_merge(self, up):
        """Called when a new merge occurs.

        May be overriden if there's an efficient way to update the data
        after a merge.

        """
        self.on_assign(up)

    def on_assign(self, up):
        """Called when a new split occurs.

        May be overriden.

        """
        for cluster in up.added:
            self.store(cluster)

    def on_cluster(self, up=None):
        """Called when the clusters change.

        Old data is kept on disk and in memory, which is useful for
        undo and redo. The `cluster_store.clean()` method can be called to
        delete the old files.

        Nothing happens during undo and redo (the data is already there).

        """
        # No need to change anything in the store if this is an undo or
        # a redo.
        if up is None or up.history is not None:
            return
        if up.description == 'merge':
            self.on_merge(up)
        elif up.description == 'assign':
            self.on_assign(up)


class FixedSizeItem(StoreItem):
    """Store data which size doesn't depend on the cluster size."""
    def load_multi(self, clusters, name):
        """Load data for several clusters."""
        if not len(clusters):
            return self.empty_values(name)
        return np.array([self.load(cluster, name)
                         for cluster in clusters])


class VariableSizeItem(StoreItem):
    """Store data which size does depend on the cluster size."""
    def load_multi(self, clusters, name, spikes=None):
        """Load data for several clusters.

        A subset of spikes can also be specified.

        """
        if not len(clusters) or (spikes is not None and not len(spikes)):
            return self.empty_values(name)
        arrays = {cluster: self.load(cluster, name)
                  for cluster in clusters}
        spc = _subset_spc(self._spikes_per_cluster, clusters)
        _assert_per_cluster_data_compatible(spc, arrays)
        pcd = PerClusterData(spc=spc,
                             arrays=arrays,
                             )
        if spikes is not None:
            pcd = pcd.subset(spike_ids=spikes)
            assert pcd.array.shape[0] == len(spikes)
        return pcd.array


#------------------------------------------------------------------------------
# Cluster store
#------------------------------------------------------------------------------

class ClusterStore(object):
    """Hold per-cluster information on disk and in memory.

    Note
    ----

    Currently, this is used to accelerate access to per-cluster data
    and statistics. All data is dynamically updated when clustering
    changes occur.

    """
    def __init__(self,
                 model=None,
                 spikes_per_cluster=None,
                 path=None,
                 ):
        self._model = model
        self._spikes_per_cluster = spikes_per_cluster
        self._memory = MemoryStore()
        self._disk = DiskStore(path) if path is not None else None
        self._items = OrderedDict()
        self._item_per_field = {}

    # Core methods
    #--------------------------------------------------------------------------

    @property
    def model(self):
        """Model."""
        return self._model

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

    def update_spikes_per_cluster(self, spikes_per_cluster):
        self._spikes_per_cluster = spikes_per_cluster
        for item in self._items.values():
            try:
                item.spikes_per_cluster = spikes_per_cluster
            except AttributeError:
                debug("Skipping set spikes_per_cluster on "
                      "store item {}.".format(item.name))

    @property
    def cluster_ids(self):
        """All cluster ids appearing in the `spikes_per_cluster` dictionary."""
        return sorted(self._spikes_per_cluster)

    # Store items
    #--------------------------------------------------------------------------

    @property
    def items(self):
        """Dictionary of registered store items."""
        return self._items

    def register_field(self, name, item_name=None):
        """Register a new piece of data to store on memory or on disk.

        Parameters
        ----------

        name : str
            The name of the field.
        item_name : str
            The name of the item.


        """
        self._item_per_field[name] = self._items[item_name]
        if self._disk:
            self._disk.register_file_extensions(name)

        # Create the load function.
        def _make_func(name):
            def load(*args, **kwargs):
                return self.load(name, *args, **kwargs)
            return load

        load = _make_func(name)

        # We create the `self.<name>()` method for loading.
        # We need to ensure that the method name isn't already attributed.
        assert not hasattr(self, name)
        setattr(self, name, load)

    def register_item(self, item_cls, **kwargs):
        """Register a `StoreItem` class in the store.

        A `StoreItem` class is responsible for storing some data to disk
        and memory. It must register one or several pieces of data.

        """
        # Instantiate the item.
        item = item_cls(cluster_store=self, **kwargs)
        assert item.fields is not None

        # Register the StoreItem instance.
        self._items[item.name] = item

        # Register all fields declared by the store item.
        for field in item.fields:
            self.register_field(field, item_name=item.name)

        return item

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
                         for item in self._items.values())
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

    def generate(self, mode=None):
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
        if hasattr(self._model, 'name'):
            name = self._model.name
        else:
            name = 'the current model'
        debug("Initializing the cluster store for {0:s}.".format(name))
        for item in self._items.values():
            item.store_all(mode)

    # Load
    #--------------------------------------------------------------------------

    def load(self, name, clusters=None, spikes=None):
        """Load some data for a number of clusters and spikes."""
        item = self._item_per_field[name]
        # Clusters requested.
        if clusters is not None:
            if _is_integer(clusters):
                # Single cluster case.
                return item.load(clusters, name)
            clusters = np.unique(clusters)
            if spikes is None:
                return item.load_multi(clusters, name)
            else:
                return item.load_multi(clusters, name, spikes=spikes)
        # Spikes requested.
        elif spikes is not None:
            assert clusters is None
            if _is_array_like(spikes):
                spikes = np.unique(spikes)
            out = item.load_spikes(spikes, name)
            assert isinstance(out, np.ndarray)
            if _is_array_like(spikes):
                assert out.shape[0] == len(spikes)
            return out
