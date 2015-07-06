# -*- coding: utf-8 -*-
from __future__ import print_function

"""Store items for Kwik."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os
import os.path as op

import numpy as np

from ...utils.selector import Selector
from ...utils.array import (_index_of,
                            _spikes_per_cluster,
                            _concatenate_per_cluster_arrays,
                            )
from ..store import ClusterStore, FixedSizeItem, VariableSizeItem


#------------------------------------------------------------------------------
# Utility functions
#------------------------------------------------------------------------------

def _default_array(shape, value=0, n_spikes=0, dtype=np.float32):
    shape = (n_spikes,) + shape[1:]
    out = np.empty(shape, dtype=dtype)
    out.fill(value)
    return out


def _atleast_nd(arr, ndim):
    if arr.ndim == ndim:
        return arr
    assert arr.ndim < ndim
    if ndim - arr.ndim == 1:
        return arr[None, ...]
    elif ndim - arr.ndim == 2:
        return arr[None, None, ...]


def _mean(arr, shape):
    if arr is not None:
        # Make sure (possibly lazy) memmapped arrays are fully loaded.
        arr = arr[...]
        assert isinstance(arr, np.ndarray)
        if arr.shape[0]:
            return arr.mean(axis=0)
    return np.zeros(shape, dtype=np.float32)


#------------------------------------------------------------------------------
# Store items
#------------------------------------------------------------------------------

class FeatureMasks(VariableSizeItem):
    """Store all features and masks of all clusters."""
    name = 'features and masks'
    fields = ['features', 'masks']

    def __init__(self, *args, **kwargs):
        # Size of the chunk used when reading features and masks from the HDF5
        # .kwx file.
        self.chunk_size = kwargs.pop('chunk_size')

        super(FeatureMasks, self).__init__(*args, **kwargs)

        self.n_features = self.model.n_features_per_channel
        self.n_channels = len(self.model.channel_order)
        self.n_spikes = self.model.n_spikes
        self.n_chunks = self.n_spikes // self.chunk_size + 1
        self._shapes['masks'] = (-1, self.n_channels)
        self._shapes['features'] = (-1, self.n_channels, self.n_features)

    def _store(self,
               cluster,
               chunk_spikes,
               chunk_spikes_per_cluster,
               chunk_features_masks,
               ):

        nc = self.n_channels
        nf = self.n_features

        # Number of spikes in the cluster and in the current
        # chunk.
        ns = len(chunk_spikes_per_cluster[cluster])

        # Find the indices of the spikes in that cluster
        # relative to the chunk.
        idx = _index_of(chunk_spikes_per_cluster[cluster], chunk_spikes)

        # Extract features and masks for that cluster, in the
        # current chunk.
        tmp = chunk_features_masks[idx, :]

        # NOTE: channel order has already been taken into account
        # by SpikeDetekt2 when saving the features and masks.
        # All we need to know here is the number of channels
        # in channel_order, there is no need to reorder.

        # Features.
        f = tmp[:, :nc * nf, 0]
        assert f.shape == (ns, nc * nf)
        f = f.ravel().astype(np.float32)

        # Masks.
        m = tmp[:, :nc * nf, 1][:, ::nf]
        assert m.shape == (ns, nc)
        m = m.ravel().astype(np.float32)

        # Save the data to disk.
        self.disk_store.store(cluster,
                              features=f,
                              masks=m,
                              append=True,
                              )

    def mean_masks(self, cluster):
        masks = self.cluster_store.masks(cluster)
        mean_masks = _mean(masks, (self.n_channels,))
        return mean_masks

    def mean_features(self, cluster):
        features = self.cluster_store.features(cluster)
        mean_features = _mean(features, (self.n_channels,))
        return mean_features

    def _store_means(self, cluster):
        if not self.disk_store:
            return
        self.disk_store.store(cluster,
                              mean_masks=self.mean_masks(cluster),
                              mean_features=self.mean_features(cluster),
                              )

    def is_consistent(self, cluster, spikes):
        """Return whether the filesizes of the two cluster store files
        (`.features` and `.masks`) are correct."""
        cluster_size = len(spikes)
        expected_file_sizes = [('masks', (cluster_size *
                                          self.n_channels *
                                          4)),
                               ('features', (cluster_size *
                                             self.n_channels *
                                             self.n_features *
                                             4))]
        for name, expected_file_size in expected_file_sizes:
            path = self.disk_store._cluster_path(cluster, name)
            if not op.exists(path):
                return False
            actual_file_size = os.stat(path).st_size
            if expected_file_size != actual_file_size:
                return False
        return True

    def store_all(self, mode=None):
        """Store the features, masks, and their means of the clusters that
        need it.

        Parameters
        ----------

        mode : str or None
            How to choose whether cluster files need to be re-generated.
            Can be one of the following options:

            * `None` or `default`: only regenerate the missing or inconsistent
              clusters
            * `force`: fully regenerate all clusters
            * `read-only`: just load the existing files, do not write anything

        """

        # No need to regenerate the cluster store if it exists and is valid.
        clusters_to_generate = self.to_generate(mode=mode)
        need_generate = len(clusters_to_generate) > 0

        if need_generate:

            self._pr.value_max = self.n_chunks + 1

            fm = self.model.features_masks
            assert fm.shape[0] == self.n_spikes

            for i in range(self.n_chunks):
                a, b = i * self.chunk_size, (i + 1) * self.chunk_size

                # Load a chunk from HDF5.
                chunk_features_masks = fm[a:b]
                assert isinstance(chunk_features_masks, np.ndarray)
                if chunk_features_masks.shape[0] == 0:
                    break

                chunk_spike_clusters = self.model.spike_clusters[a:b]
                chunk_spikes = np.arange(a, b)

                # Split the spikes.
                chunk_spc = _spikes_per_cluster(chunk_spikes,
                                                chunk_spike_clusters)

                # Go through the clusters appearing in the chunk and that
                # need to be re-generated.
                clusters = (set(chunk_spc.keys()).
                            intersection(set(clusters_to_generate)))
                for cluster in sorted(clusters):
                    self._store(cluster,
                                chunk_spikes,
                                chunk_spc,
                                chunk_features_masks,
                                )

                # Update the progress reporter.
                self._pr.value += 1

            # Store mean features and masks on disk.
            self._pr.value = 0
            self._pr.value_max = len(clusters_to_generate)
            for cluster in clusters_to_generate:
                self._store_means(cluster)
                self._pr.value += 1

        self._pr.set_complete()

    def load(self, cluster, name):
        """Load features or masks for a cluster.

        This uses the cluster store if possible, otherwise it falls back
        to the model (much slower).

        """
        assert name in ('features', 'masks')
        dtype = np.float32
        shape = self._shapes[name]
        if self.disk_store:
            data = self.disk_store.load(cluster, name, dtype, shape)
            if data is not None:
                return data
        # Fallback to load_spikes if the data could not be obtained from
        # the store.
        spikes = self.spikes_per_cluster[cluster]
        return self.load_spikes(spikes, name)

    def load_spikes(self, spikes, name):
        """Load features or masks for an array of spikes."""
        assert name in ('features', 'masks')
        shape = self._shapes[name]
        data = getattr(self.model, name)
        if data is not None and (isinstance(spikes, slice) or len(spikes)):
            out = data[spikes]
        # Default masks and features.
        else:
            out = _default_array(shape,
                                 value=0. if name == 'features' else 1.,
                                 n_spikes=len(spikes),
                                 )
        return out.reshape(shape)

    def on_merge(self, up):
        """Create the cluster store files of the merged cluster
        from the files of the old clusters.

        This is basically a concatenation of arrays, but the spike order
        needs to be taken into account.

        """
        if not self.disk_store:
            return
        clusters = up.deleted
        spc = up.old_spikes_per_cluster
        # We load all masks and features of the merged clusters.
        for name, shape in [('features',
                             (-1, self.n_channels, self.n_features)),
                            ('masks',
                             (-1, self.n_channels)),
                            ]:
            arrays = {cluster: self.disk_store.load(cluster,
                                                    name,
                                                    dtype=np.float32,
                                                    shape=shape)
                      for cluster in clusters}
            # Then, we concatenate them using the right insertion order
            # as defined by the spikes.

            # OPTIM: this could be made a bit faster by passing
            # both arrays at once.
            concat = _concatenate_per_cluster_arrays(spc, arrays)

            # Finally, we store the result into the new cluster.
            self.disk_store.store(up.added[0], **{name: concat})

    def on_assign(self, up):
        """Create the cluster store files of the new clusters
        from the files of the old clusters.

        The files of all old clusters are loaded, re-split and concatenated
        to form the new cluster files.

        """
        if not self.disk_store:
            return
        for name, shape in [('features',
                             (-1, self.n_channels, self.n_features)),
                            ('masks',
                             (-1, self.n_channels)),
                            ]:
            # Load all data from the old clusters.
            old_arrays = {cluster: self.disk_store.load(cluster,
                                                        name,
                                                        dtype=np.float32,
                                                        shape=shape)
                          for cluster in up.deleted}
            # Create the new arrays.
            for new in up.added:
                # Find the old clusters which are parents of the current
                # new cluster.
                old_clusters = [o
                                for (o, n) in up.descendants
                                if n == new]
                # Spikes per old cluster, used to create
                # the concatenated array.
                spc = {}
                old_arrays_sub = {}
                # Find the relative spike indices of every old cluster
                # for the current new cluster.
                for old in old_clusters:
                    # Find the spike indices in the old and new cluster.
                    old_spikes = up.old_spikes_per_cluster[old]
                    new_spikes = up.new_spikes_per_cluster[new]
                    old_in_new = np.in1d(old_spikes, new_spikes)
                    old_spikes_subset = old_spikes[old_in_new]
                    spc[old] = old_spikes_subset
                    # Extract the data from the old cluster to
                    # be moved to the new cluster.
                    old_spikes_rel = _index_of(old_spikes_subset,
                                               old_spikes)
                    old_arrays_sub[old] = old_arrays[old][old_spikes_rel]
                # Construct the array of the new cluster.
                concat = _concatenate_per_cluster_arrays(spc,
                                                         old_arrays_sub)
                # Save it in the cluster store.
                self.disk_store.store(new, **{name: concat})

    def on_cluster(self, up=None):
        super(FeatureMasks, self).on_cluster(up)
        # No need to change anything in the store if this is an undo or
        # a redo.
        if up is None or up.history is not None:
            return
        # Store the means of the new clusters.
        for cluster in up.added:
            self._store_means(cluster)


class Waveforms(VariableSizeItem):
    """A cluster store item that manages the waveforms of all clusters."""
    name = 'waveforms'
    fields = ['waveforms']

    def __init__(self, *args, **kwargs):
        self.n_spikes_max = kwargs.pop('n_spikes_max')
        self.excerpt_size = kwargs.pop('excerpt_size')
        # Size of the chunk used when reading waveforms from the raw data.
        self.chunk_size = kwargs.pop('chunk_size')

        super(Waveforms, self).__init__(*args, **kwargs)

        self.n_channels = len(self.model.channel_order)
        self.n_samples = self.model.n_samples_waveforms
        self.n_spikes = self.model.n_spikes

        self._shapes['waveforms'] = (-1, self.n_samples, self.n_channels)
        self._selector = Selector(self.model.spike_clusters,
                                  n_spikes_max=self.n_spikes_max,
                                  excerpt_size=self.excerpt_size,
                                  )

        # Get or create the subset spikes per cluster dictionary.
        spc = (self.disk_store.load_file('waveforms_spikes')
               if self.disk_store else None)
        if spc is None:
            spc = self._selector.subset_spikes_clusters(self.model.cluster_ids)
            if self.disk_store:
                self.disk_store.save_file('waveforms_spikes', spc)
        self._spikes_per_cluster = spc

    def _subset_spikes_cluster(self, cluster, force=False):
        if force or cluster not in self._spikes_per_cluster:
            spikes = self._selector.subset_spikes_clusters([cluster])[cluster]
            # Persist the new _spikes_per_cluster array on disk.
            self._spikes_per_cluster[cluster] = spikes
            if self.disk_store:
                self.disk_store.save_file('waveforms_spikes',
                                          self._spikes_per_cluster)
        return self._spikes_per_cluster[cluster]

    @property
    def spikes_per_cluster(self):
        """Spikes per cluster."""
        # WARNING: this is read-only, because this item is responsible
        # for the spike subselection.
        return self._spikes_per_cluster

    def _store_mean(self, cluster):
        if not self.disk_store:
            return
        self.disk_store.store(cluster,
                              mean_waveforms=self.mean_waveforms(cluster),
                              )

    def waveforms_and_mean(self, cluster):
        spikes = self._subset_spikes_cluster(cluster, force=True)
        waveforms = self.model.waveforms[spikes].astype(np.float32)
        mean_waveforms = _mean(waveforms, (self.n_samples, self.n_channels))
        return waveforms, mean_waveforms

    def mean_waveforms(self, cluster):
        return self.waveforms_and_mean(cluster)[1]

    def store(self, cluster):
        """Store waveforms and mean waveforms."""
        if not self.disk_store:
            return
        # NOTE: make sure to erase old spikes for that cluster.
        # Typical case merge, undo, different merge.
        waveforms, mean_waveforms = self.waveforms_and_mean(cluster)
        self.disk_store.store(cluster,
                              waveforms=waveforms,
                              mean_waveforms=mean_waveforms,
                              )

    def store_all(self, mode=None):
        if not self.disk_store:
            return

        clusters_to_generate = self.to_generate(mode=mode)
        need_generate = len(clusters_to_generate) > 0

        if need_generate:
            spc = {cluster: self._subset_spikes_cluster(cluster, force=True)
                   for cluster in clusters_to_generate}

            # All spikes to fetch and save in the store.
            spike_ids = _concatenate_per_cluster_arrays(spc, spc)

            # Load waveforms chunk by chunk for I/O contiguity.
            n_chunks = len(spike_ids) // self.chunk_size + 1
            self._pr.value_max = n_chunks + 1

            for i in range(n_chunks):
                a, b = i * self.chunk_size, (i + 1) * self.chunk_size
                spk = spike_ids[a:b]

                # Load a chunk of waveforms.
                chunk_waveforms = self.model.waveforms[spk]
                assert isinstance(chunk_waveforms, np.ndarray)
                if chunk_waveforms.shape[0] == 0:
                    break

                chunk_spike_clusters = self.model.spike_clusters[spk]

                # Split the spikes.
                chunk_spc = _spikes_per_cluster(spk, chunk_spike_clusters)

                # Go through the clusters appearing in the chunk and that
                # need to be re-generated.
                clusters = (set(chunk_spc.keys()).
                            intersection(set(clusters_to_generate)))
                for cluster in sorted(clusters):
                    i = _index_of(chunk_spc[cluster], spk)
                    w = chunk_waveforms[i].astype(np.float32)
                    self.disk_store.store(cluster,
                                          waveforms=w,
                                          append=True
                                          )

                self._pr.increment()

            # Store mean waveforms on disk.
            self._pr.value = 0
            self._pr.value_max = len(clusters_to_generate)
            for cluster in clusters_to_generate:
                self._store_mean(cluster)
                self._pr.value += 1

    def is_consistent(self, cluster, spikes):
        """Return whether the waveforms and spikes match."""
        path_w = self.disk_store._cluster_path(cluster, 'waveforms')
        if not op.exists(path_w):
            return False
        file_size_w = os.stat(path_w).st_size
        n_spikes_w = file_size_w // (self.n_channels * self.n_samples * 4)
        if n_spikes_w != len(self.spikes_per_cluster[cluster]):
            return False
        return True

    def load(self, cluster, name='waveforms'):
        """Load waveforms for a cluster.

        This uses the cluster store if possible, otherwise it falls back
        to the model (much slower).

        """
        assert name == 'waveforms'
        dtype = np.float32
        if self.disk_store:
            data = self.disk_store.load(cluster,
                                        name,
                                        dtype,
                                        self._shapes[name],
                                        )
            if data is not None:
                return data
        # Fallback to load_spikes if the data could not be obtained from
        # the store.
        spikes = self._subset_spikes_cluster(cluster)
        return self.load_spikes(spikes, name)

    def load_spikes(self, spikes, name):
        """Load waveforms for an array of spikes."""
        assert name == 'waveforms'
        data = getattr(self.model, name)
        shape = self._shapes[name]
        if data is not None and (isinstance(spikes, slice) or len(spikes)):
            return data[spikes]
        # Default waveforms.
        return _default_array(shape, value=0., n_spikes=len(spikes))


class ClusterStatistics(FixedSizeItem):
    """Manage cluster statistics."""
    name = 'statistics'

    def __init__(self, *args, **kwargs):
        super(ClusterStatistics, self).__init__(*args, **kwargs)
        self._funcs = {}
        self.n_channels = len(self.model.channel_order)
        self.n_samples_waveforms = self.model.n_samples_waveforms
        self.n_features = self.model.n_features_per_channel
        self._shapes = {
            'mean_masks': (-1, self.n_channels),
            'mean_features': (-1, self.n_channels, self.n_features),
            'mean_waveforms': (-1, self.n_samples_waveforms, self.n_channels),
            'mean_probe_position': (-1, 2),
            'main_channels': (-1, self.n_channels),
            'n_unmasked_channels': (-1,),
            'n_spikes': (-1,),
        }
        self.fields = list(self._shapes.keys())

    def add(self, name, func, shape):
        """Add a new statistics."""
        self.fields.append(name)
        self._funcs[name] = func
        self._shapes[name] = shape

    def remove(self, name):
        """Remove a statistics."""
        self.fields.remove(name)
        del self.funcs[name]

    def _load_mean(self, cluster, name):
        if self.disk_store:
            # Load from the disk store if possible.
            return self.disk_store.load(cluster, name, np.float32,
                                        self._shapes[name][1:])
        else:
            # Otherwise compute the mean directly from the model.
            item_name = ('waveforms'
                         if name == 'mean_waveforms'
                         else 'features and masks')
            item = self.cluster_store.items[item_name]
            return getattr(item, name)(cluster)

    def mean_masks(self, cluster):
        return self._load_mean(cluster, 'mean_masks')

    def mean_features(self, cluster):
        return self._load_mean(cluster, 'mean_features')

    def mean_waveforms(self, cluster):
        return self._load_mean(cluster, 'mean_waveforms')

    def unmasked_channels(self, cluster):
        mean_masks = self.load(cluster, 'mean_masks')
        return np.nonzero(mean_masks > .1)[0]

    def n_unmasked_channels(self, cluster):
        unmasked_channels = self.load(cluster, 'unmasked_channels')
        return len(unmasked_channels)

    def mean_probe_position(self, cluster):
        mean_masks = self.load(cluster, 'mean_masks')
        if mean_masks is not None and mean_masks.shape[0]:
            mean_cluster_position = (np.sum(self.model.probe.positions *
                                     mean_masks[:, np.newaxis], axis=0) /
                                     max(1, np.sum(mean_masks)))
        else:
            mean_cluster_position = np.zeros((2,), dtype=np.float32)

        return mean_cluster_position

    def n_spikes(self, cluster):
        return len(self._spikes_per_cluster[cluster])

    def main_channels(self, cluster):
        mean_masks = self.load(cluster, 'mean_masks')
        unmasked_channels = self.load(cluster, 'unmasked_channels')
        # Weighted mean of the channels, weighted by the mean masks.
        main_channels = np.argsort(mean_masks)[::-1]
        main_channels = np.array([c for c in main_channels
                                  if c in unmasked_channels])
        return main_channels

    def store_default(self, cluster):
        """Compute the built-in statistics for one cluster.

        The mean masks, features, and waveforms are loaded from disk.

        """
        self.memory_store.store(cluster,
                                mean_masks=self.mean_masks(cluster),
                                mean_features=self.mean_features(cluster),
                                mean_waveforms=self.mean_waveforms(cluster),
                                )

        n_spikes = self.n_spikes(cluster)

        # Note: some of the default statistics below rely on other statistics
        # computed previously and stored in the memory store.
        unmasked_channels = self.unmasked_channels(cluster)
        self.memory_store.store(cluster,
                                unmasked_channels=unmasked_channels)

        n_unmasked_channels = self.n_unmasked_channels(cluster)
        self.memory_store.store(cluster,
                                n_unmasked_channels=n_unmasked_channels)

        mean_probe_position = self.mean_probe_position(cluster)
        main_channels = self.main_channels(cluster)
        n_unmasked_channels = self.n_unmasked_channels(cluster)
        self.memory_store.store(cluster,
                                mean_probe_position=mean_probe_position,
                                main_channels=main_channels,
                                n_unmasked_channels=n_unmasked_channels,
                                n_spikes=n_spikes,
                                )

    def store(self, cluster, name=None):
        """Compute all statistics for one cluster."""
        if name is None:
            self.store_default(cluster)
            for func in self._funcs.values():
                func(cluster)
        else:
            assert name in self._funcs
            self._funcs[name](cluster)

    def load(self, cluster, name):
        """Return a cluster statistic."""
        if cluster in self.memory_store:
            return self.memory_store.load(cluster, name)
        else:
            # If the item hadn't been stored, compute it here by calling
            # the corresponding method.
            if hasattr(self, name):
                return getattr(self, name)(cluster)
            # Custom statistic.
            else:
                return self._funcs[name](cluster)

    def is_consistent(self, cluster, spikes):
        """Return whether a cluster is consistent."""
        return cluster in self.memory_store


#------------------------------------------------------------------------------
# Store creation
#------------------------------------------------------------------------------

def create_store(model,
                 spikes_per_cluster=None,
                 path=None,
                 features_masks_chunk_size=100000,
                 waveforms_n_spikes_max=None,
                 waveforms_excerpt_size=None,
                 waveforms_chunk_size=1000,
                 ):
    """Create a cluster store for a model."""
    assert spikes_per_cluster is not None
    cluster_store = ClusterStore(model=model,
                                 spikes_per_cluster=spikes_per_cluster,
                                 path=path,
                                 )

    # Create the FeatureMasks store item.
    # chunk_size is the number of spikes to load at once from
    # the features_masks array.
    cluster_store.register_item(FeatureMasks,
                                chunk_size=features_masks_chunk_size,
                                )
    cluster_store.register_item(Waveforms,
                                n_spikes_max=waveforms_n_spikes_max,
                                excerpt_size=waveforms_excerpt_size,
                                chunk_size=waveforms_chunk_size,
                                )
    cluster_store.register_item(ClusterStatistics)
    return cluster_store
