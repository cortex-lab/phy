# -*- coding: utf-8 -*-
from __future__ import print_function

"""Session structure."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os
import os.path as op
import shutil
from functools import partial

import numpy as np

from ...ext.six import string_types
from ...utils._misc import _ensure_path_exists
from ...utils.array import _index_of
from ...utils.event import EventEmitter
from ...utils.logging import info
from ...utils.settings import SettingsManager, declare_namespace
from ...io.kwik_model import KwikModel
from ._history import GlobalHistory
from .clustering import Clustering
from ._utils import _spikes_per_cluster, _concatenate_per_cluster_arrays
from .selector import Selector
from .store import ClusterStore, StoreItem
from .view_model import (WaveformViewModel,
                         FeatureViewModel,
                         CorrelogramViewModel,
                         )
from .wizard import Wizard, _best_clusters


#------------------------------------------------------------------------------
# BaseSession class
#------------------------------------------------------------------------------

class BaseSession(EventEmitter):
    """Provide actions, views, and an event system for creating an interactive
    session."""
    def __init__(self):
        super(BaseSession, self).__init__()
        self._actions = []

    def action(self, func=None, title=None):
        """Decorator for a callback function of an action.

        The 'title' argument is used as a title for the GUI button.

        """
        if func is None:
            return partial(self.action, title=title)

        # HACK: handle the case where the first argument is the title.
        if isinstance(func, string_types):
            return partial(self.action, title=func)

        # Register the action.
        self._actions.append({'func': func, 'title': title})

        # Set the action function as a Session method.
        setattr(self, func.__name__, func)

        return func

    @property
    def actions(self):
        """List of registered actions."""
        return self._actions

    def execute_action(self, action, *args, **kwargs):
        """Execute an action defined by an item in the 'actions' list."""
        action['func'](*args, **kwargs)


#------------------------------------------------------------------------------
# Store items
#------------------------------------------------------------------------------

class FeatureMasks(StoreItem):
    name = 'features and masks'
    fields = [('features', 'disk', np.float32,),
              ('masks', 'disk', np.float32,),
              ('mean_masks', 'memory'),
              ('sum_masks', 'memory'),
              ('n_unmasked_channels', 'memory'),
              ('main_channels', 'memory'),
              ('mean_probe_position', 'memory'),
              ]
    chunk_size = None

    def __init__(self, *args, **kwargs):
        super(FeatureMasks, self).__init__(*args, **kwargs)

        self.n_features = self.model.n_features_per_channel
        self.n_channels = len(self.model.channel_order)
        self.n_spikes = self.model.n_spikes
        self.n_chunks = self.n_spikes // self.chunk_size + 1

        # Set the shape of the features and masks.
        self.fields[0] = ('features', 'disk',
                          np.float32, (-1, self.n_channels, self.n_features))
        self.fields[1] = ('masks', 'disk',
                          np.float32, (-1, self.n_channels))

        self.progress_reporter.set_max(features_masks=self.n_chunks)

    def _need_generate(self, cluster_sizes):
        """Return whether the whole cluster store needs to be
        re-generated or not."""
        for cluster in sorted(cluster_sizes):
            cluster_size = cluster_sizes[cluster]
            expected_file_size = (cluster_size * self.n_channels * 4)
            path = self.disk_store._cluster_path(cluster, 'masks')
            # If a file is missing, need to re-generate.
            if not op.exists(path):
                return True
            actual_file_size = os.stat(path).st_size
            # If a file size is incorrect, need to re-generate.
            if expected_file_size != actual_file_size:
                return True
        return False

    def _store_extra_fields(self, clusters):
        """Store all extra mask fields."""

        self.progress_reporter.set_max(masks_extra=len(clusters))

        for cluster in clusters:

            # Load the masks.
            masks = self.disk_store.load(cluster, 'masks',
                                         dtype=np.float32,
                                         shape=(-1, self.n_channels))
            assert isinstance(masks, np.ndarray)

            # Extra fields.
            sum_masks = masks.sum(axis=0)
            mean_masks = sum_masks / float(masks.shape[0])
            unmasked_channels = np.nonzero(mean_masks > .1)[0]
            n_unmasked_channels = len(unmasked_channels)
            # Weighted mean of the channels, weighted by the mean masks.
            mean_probe_position = (self.model.probe.positions *
                                   mean_masks[:, np.newaxis]).mean(axis=0)
            main_channels = np.intersect1d(np.argsort(mean_masks)[::-1],
                                           unmasked_channels)
            self.memory_store.store(cluster,
                                    mean_masks=mean_masks,
                                    sum_masks=sum_masks,
                                    n_unmasked_channels=n_unmasked_channels,
                                    mean_probe_position=mean_probe_position,
                                    main_channels=main_channels,
                                    )

            # Update the progress reporter.
            self.progress_reporter.increment('masks_extra')

    def _store_cluster(self,
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
        # by SpikeDetekt2 when saving the features and wavforms.
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

    def store_all_clusters(self, spikes_per_cluster):
        """Initialize all cluster files, loop over all spikes, and
        copy the data."""
        cluster_sizes = {cluster: len(spikes)
                         for cluster, spikes in spikes_per_cluster.items()}
        clusters = sorted(spikes_per_cluster)

        # No need to regenerate the cluster store if it exists and is valid.
        need_generate = self._need_generate(cluster_sizes)
        if need_generate:

            self.progress_reporter.set(features_masks=0)

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

                # Go through the clusters appearing in the chunk.
                for cluster in sorted(chunk_spc.keys()):
                    self._store_cluster(cluster,
                                        chunk_spikes,
                                        chunk_spc,
                                        chunk_features_masks,
                                        )

                # Update the progress reporter.
                self.progress_reporter.increment('features_masks')

        # Store extra fields from the masks.
        self._store_extra_fields(clusters)

        self.progress_reporter.set_complete()

    def _merge(self, up):
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

    def _assign(self, up):
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
        # No need to change anything in the store if this is an undo or
        # a redo.
        if up is None or up.history is not None:
            return
        if up.description == 'merge':
            self._merge(up)
        elif up.description == 'assign':
            self._assign(up)
        # Compute the extra fields for the new clusters.
        self._store_extra_fields(up.added)


#------------------------------------------------------------------------------
# Session class
#------------------------------------------------------------------------------

def _process_ups(ups):
    """This function processes the UpdateInfo instances of the two
    undo stacks (clustering and cluster metadata) and concatenates them
    into a single UpdateInfo instance."""
    if len(ups) == 0:
        return
    elif len(ups) == 1:
        return ups[0]
    elif len(ups) == 2:
        up = ups[0]
        up.update(ups[1])
        return up
    else:
        raise NotImplementedError()


_VIEW_MODELS = {
    'waveforms': WaveformViewModel,
    'features': FeatureViewModel,
    'correlograms': CorrelogramViewModel,
}


class Session(BaseSession):
    """A manual clustering session."""
    def __init__(self, phy_user_dir=None):
        super(Session, self).__init__()
        self.model = None
        self.phy_user_dir = phy_user_dir

        # Instantiate the SettingsManager which manages
        # the settings files.
        self.settings_manager = SettingsManager(phy_user_dir)

        # self.action and self.connect are decorators.
        self.action(self.open, title='Open')
        self.action(self.close, title='Close')
        self.action(self.select, title='Select clusters')
        self.action(self.merge, title='Merge')
        self.action(self.split, title='Split')
        self.action(self.move, title='Move clusters to a group')
        self.action(self.undo, title='Undo')
        self.action(self.redo, title='Redo')

        self.connect(self.on_open)
        self.connect(self.on_cluster)
        self.connect(self.on_close)

    # Settings
    # -------------------------------------------------------------------------

    def get_user_settings(self, key):
        return self.settings_manager.get_user_settings(key,
                                                       scope='experiment')

    def set_user_settings(self, key=None, value=None,
                          path=None, file_namespace=None):
        return self.settings_manager.set_user_settings(
            key, value, scope='experiment', path=path,
            file_namespace=file_namespace)

    def get_internal_settings(self, key):
        return self.settings_manager.get_internal_settings(key,
                                                           scope='experiment',
                                                           )

    def set_internal_settings(self, key, value):
        return self.settings_manager.set_internal_settings(key,
                                                           value,
                                                           scope='experiment',
                                                           )

    def _load_default_settings(self):
        """Load default settings for manual clustering."""
        curdir = op.dirname(op.realpath(__file__))
        # This is a namespace available in the config file.
        file_namespace = {
            'n_spikes': self.model.n_spikes,
            'n_channels': self.model.n_channels,
        }
        declare_namespace('manual_clustering')
        self.set_user_settings(path=op.join(curdir, 'default_settings.py'),
                               file_namespace=file_namespace)

    # File-related actions
    # -------------------------------------------------------------------------

    def _backup_kwik(self, filename):
        """Save a copy of the Kwik file before opening it."""
        backup_filename = filename + '.bak'
        if not op.exists(backup_filename):
            info("Saving a backup of the Kwik file "
                 "in {0}.".format(backup_filename))
            shutil.copyfile(filename, backup_filename)

    def open(self, filename=None, model=None):
        if filename is not None:
            self._backup_kwik(filename)
        if model is None:
            model = KwikModel(filename)
        self.model = model
        self.experiment_path = (op.realpath(filename)
                                if filename else self.phy_user_dir)
        self.experiment_dir = op.dirname(self.experiment_path)
        self.experiment_name = model.name
        self.emit('open')

    def save(self):
        """Save the spike clusters and cluster groups to the Kwik file."""
        groups = {cluster: self.cluster_metadata.group(cluster)
                  for cluster in self.clustering.cluster_ids}
        self.model.save(self.clustering.spike_clusters,
                        groups)
        info("Saved {0:s}.".format(self.model.filename))

    def close(self):
        self.emit('close')
        self.model = None
        self.experiment_path = None
        self.experiment_dir = None

    # Clustering actions
    # -------------------------------------------------------------------------

    def select(self, clusters):
        self.selector.selected_clusters = clusters
        self.emit('select', self.selector)

    def merge(self, clusters):
        up = self.clustering.merge(clusters)
        self.emit('cluster', up=up)

    def split(self, spikes):
        up = self.clustering.split(spikes)
        self.emit('cluster', up=up)

    def move(self, clusters, group):
        up = self.cluster_metadata.set_group(clusters, group)
        self.emit('cluster', up=up)

    def undo(self):
        up = self._global_history.undo()
        self.emit('cluster', up=up, add_to_stack=False)

    def redo(self):
        up = self._global_history.redo()
        self.emit('cluster', up=up, add_to_stack=False)

    # Properties
    # -------------------------------------------------------------------------

    @property
    def clusters(self):
        return self.clustering.cluster_ids

    # Event callbacks
    # -------------------------------------------------------------------------

    def _create_cluster_store(self):

        # Kwik store in experiment_dir/name.phy/cluster_store.
        store_path = op.join(self.settings_manager.phy_experiment_dir,
                             'cluster_store',
                             str(self.model.channel_group),
                             self.model.clustering
                             )
        _ensure_path_exists(store_path)

        # Instantiate the store.
        self.cluster_store = ClusterStore(model=self.model,
                                          path=store_path,
                                          )

        # chunk_size is the number of spikes to load at once from
        # the features_masks array.
        cs = self.get_user_settings('manual_clustering.'
                                    'store_chunk_size') or 100000
        FeatureMasks.chunk_size = cs
        self.cluster_store.register_item(FeatureMasks)

        @self.cluster_store.progress_reporter.connect
        def on_report(value, value_max):
            print("Initializing the cluster store: "
                  "{0:.2f}%.".format(100 * value / float(value_max)),
                  end='\r')

        # Generate the cluster store if it doesn't exist or is invalid.
        # If the cluster store already exists and is consistent
        # with the data, it is not recreated.
        self.cluster_store.generate(self.clustering.spikes_per_cluster)

        @self.connect
        def on_cluster(up=None, add_to_stack=None):
            self.cluster_store.on_cluster(up)

    def on_open(self):
        """Update the session after new data has been loaded.

        TODO: call this after the channel groups has changed.

        """

        # Load the default settings for manual clustering.
        self._load_default_settings()

        # Load all experiment settings.
        self.settings_manager.set_experiment_path(self.experiment_path)

        # Create the history.
        self._global_history = GlobalHistory(process_ups=_process_ups)

        # Create the Clustering instance.
        spike_clusters = self.model.spike_clusters
        self.clustering = Clustering(spike_clusters)

        # Create the Selector instance.
        self.selector = Selector(spike_clusters)
        self.cluster_metadata = self.model.cluster_metadata

        # Create the cluster store.
        self._create_cluster_store()

        # Create the wizard.
        self.wizard = Wizard(cluster_metadata=self.cluster_metadata)
        self.wizard.cluster_ids = self.clustering.cluster_ids

        # Set the similarity and quality functions for the wizard.
        @self.wizard.set_similarity
        def similarity(target, candidate):
            """Compute the dot product between the mean masks of
            two clusters."""
            return np.dot(self.cluster_store.mean_masks(target),
                          self.cluster_store.mean_masks(candidate))

        @self.wizard.set_quality
        def quality(cluster):
            """Return the maximum mean_masks across all channels
            for a given cluster."""
            return self.cluster_store.mean_masks(cluster).max()

    def on_close(self):
        self.settings_manager.save()

    def on_cluster(self, up=None, add_to_stack=True):
        # Update the wizard.
        self.wizard.cluster_ids = self.clustering.cluster_ids

        # Update the global history.
        if add_to_stack and up is not None:
            if up.description.startswith('metadata'):
                self._global_history.action(self.cluster_metadata)
            elif up.description in ('merge', 'assign'):
                self._global_history.action(self.clustering)

    # Wizard
    # -------------------------------------------------------------------------

    def best_clusters(self, quality=None, n_max=None):
        """Return the best clusters by decreasing order of quality,
        for a given 'cluster => quality' function. By default,
        this uses the quality function used in the wizard."""
        if quality is None:
            return self.wizard.best_clusters(n_max=n_max)
        else:
            return _best_clusters(self.clusters, quality, n_max=n_max)

    def most_similar_clusters(self, cluster=None, n_max=None):
        """Return the most similar clusters to a given cluster (the current
        best cluster by default), by decreasing order of similarity."""
        return self.wizard.most_similar_clusters(cluster=cluster, n_max=n_max)

    # Show views
    # -------------------------------------------------------------------------

    def _create_view(self, view_model, show=True):
        view = view_model.view
        view_name = view_model.view_name

        @self.connect
        def on_open():
            if self.model is None:
                return
            view_model.on_open()
            view.update()

        @self.connect
        def on_cluster(up=None):
            view_model.on_cluster(up)
            view.update()

        @self.connect
        def on_select(selector):
            if len(selector.selected_clusters) == 0:
                return
            if view.visual.empty:
                on_open()

            n_spikes_max = self.get_user_settings('manual_clustering.' +
                                                  view_name +
                                                  '_n_spikes_max')
            excerpt_size = self.get_user_settings('manual_clustering.' +
                                                  view_name +
                                                  '_excerpt_size')
            spikes = selector.subset_spikes(n_spikes_max=n_spikes_max,
                                            excerpt_size=excerpt_size)
            view_model.on_select(selector.selected_clusters,
                                 spikes)
            view.update()

        # Unregister the callbacks when the view is closed.
        @view.connect
        def on_close(event):
            self.unconnect(on_open, on_cluster, on_select)

        # Make sure the view is correctly initialized when it is created
        # *after* that the data has been loaded.
        @view.connect
        def on_draw(event):
            if view.visual.empty:
                on_open()
                on_select(self.selector)

        if show:
            view.show()

        return view

    def _create_view_model(self, name, **kwargs):
        vm_class = _VIEW_MODELS[name]
        return vm_class(self.model, store=self.cluster_store, **kwargs)

    def show_waveforms(self):
        """Show a WaveformView and return a ViewModel instance."""

        # Persist scale factor.
        sf_name = 'manual_clustering.waveforms_scale_factor'
        sf = self.get_internal_settings(sf_name) or .01
        vm = self._create_view_model('waveforms', scale_factor=sf)

        self._create_view(vm)

        @vm.view.connect
        def on_draw(event):
            # OPTIM: put this when the model or the view is closed instead
            # No need to run this at every draw!
            sf = vm.view.box_scale[1] / vm.view.visual.default_box_scale[1]
            sf = sf * vm.scale_factor
            self.set_internal_settings(sf_name, sf)

        return vm

    def show_features(self):
        """Show a FeatureView and return a ViewModel instance."""

        # Persist scale factor.
        sf_name = 'manual_clustering.features_scale_factor'
        sf = self.get_internal_settings(sf_name) or .01
        vm = self._create_view_model('features', scale_factor=sf)

        self._create_view(vm)

        @vm.view.connect
        def on_draw(event):
            self.set_internal_settings(sf_name,
                                       vm.view.zoom * vm.scale_factor)

        return vm

    def show_correlograms(self):
        """Show a CorrelogramView and return a ViewModel instance."""
        args = 'binsize', 'winsize_bins', 'n_excerpts', 'excerpt_size'
        kwargs = {k: self.get_user_settings('manual_clustering.'
                                            'correlograms_' + k)
                  for k in args}
        vm = self._create_view_model('correlograms', **kwargs)
        self._create_view(vm)
        return vm
