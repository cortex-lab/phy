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
from ...utils.array import _index_of, _is_array_like
from ...utils.event import EventEmitter, ProgressReporter
from ...utils.logging import info
from ...utils.settings import SettingsManager, declare_namespace
from ...io.kwik_model import KwikModel
from ._history import GlobalHistory
from .clustering import Clustering
from ._utils import (_spikes_per_cluster,
                     _concatenate_per_cluster_arrays,
                     _update_cluster_selection,
                     )
from .store import ClusterStore, StoreItem
from .view_model import (WaveformViewModel,
                         FeatureViewModel,
                         CorrelogramViewModel,
                         TraceViewModel,
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
    """A cluster store item that manages the features and masks of
    all clusters."""
    name = 'features and masks'
    fields = [('features', 'disk', np.float32,),
              ('masks', 'disk', np.float32,),
              # The following fields are some basic cluster statistics
              # used in the library.
              ('mean_masks', 'memory'),
              ('sum_masks', 'memory'),
              ('n_unmasked_channels', 'memory'),
              ('main_channels', 'memory'),
              ('mean_probe_position', 'memory'),
              ]

    # Size of the chunk used when reading features and masks from the HDF5
    # .kwx file.
    chunk_size = None

    def __init__(self, *args, **kwargs):
        self._pr_disk = kwargs.pop('progress_reporter_disk')
        self._pr_memory = kwargs.pop('progress_reporter_memory')
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

    def _store_extra_fields(self, clusters):
        """Store all extra mask fields."""

        self._pr_memory.value_max = len(clusters)

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
            main_channels = np.argsort(mean_masks)[::-1]
            main_channels = np.array([c for c in main_channels
                                      if c in unmasked_channels])
            self.memory_store.store(cluster,
                                    mean_masks=mean_masks,
                                    sum_masks=sum_masks,
                                    n_unmasked_channels=n_unmasked_channels,
                                    mean_probe_position=mean_probe_position,
                                    main_channels=main_channels,
                                    )

            # Update the progress reporter.
            self._pr_memory.value += 1

        self._pr_memory.set_complete()

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

    def store_all_clusters(self, mode=None):
        """Store the features and masks of the clusters that need to be
        regenerated.

        Parameters
        ----------

        mode : str or None
            How to choose whether cluster files need to be re-generated.
            Can be one of the following options:

            * None or 'default': only regenerate the missing or inconsistent
              clusters
            * 'force': fully regenerate all clusters
            * 'read-only': just load the existing files, do not write anything

        """

        # No need to regenerate the cluster store if it exists and is valid.
        clusters_to_generate = self.to_generate(mode=mode)
        need_generate = len(clusters_to_generate) > 0

        if need_generate:

            self._pr_disk.value_max = self.n_chunks

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
                    self._store_cluster(cluster,
                                        chunk_spikes,
                                        chunk_spc,
                                        chunk_features_masks,
                                        )

                # Update the progress reporter.
                self._pr_disk.value += 1

        self._pr_disk.set_complete()

        # Store extra fields from the masks.
        self._store_extra_fields(self.cluster_ids)

    def _merge(self, up):
        """Create the cluster store files of the merged cluster
        from the files of the old clusters.

        This is basically a concatenation of arrays, but the spike order
        needs to be taken into account.

        """
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
        """Create the cluster store files of the new clusters
        from the files of the old clusters.

        The files of all old clusters are loaded, re-split and concatenated
        to form the new cluster files.

        """
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
        """Generate the `.features` and `.masks` files of the newly-created
        clusters, and compute their cluster statistics.

        Old data is kept on disk and in memory, which is useful for
        undo and redo. The `cluster_store.clean()` method can be called to
        delete the old files.

        """
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
    'traces': TraceViewModel,
}


class Session(BaseSession):
    """A manual clustering session.

    This is the main object used for manual clustering. It implements
    all common actions:

    * Loading a dataset (.kwik file)
    * Listing the clusters
    * Changing the current channel group or current clustering
    * Showing views (waveforms, features, correlograms, etc.)
    * Clustering actions: merge, split, undo, redo
    * Wizard: cluster quality, best clusters, most similar clusters
    * Save back to .kwik

    """
    def __init__(self, phy_user_dir=None):
        super(Session, self).__init__()
        self.model = None
        self.phy_user_dir = phy_user_dir
        self._selected_clusters = []

        # Instantiate the SettingsManager which manages
        # the settings files.
        self.settings_manager = SettingsManager(phy_user_dir)
        self._load_default_settings()

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
        """Load a user settings."""
        return self.settings_manager.get_user_settings(key,
                                                       scope='experiment')

    def load_user_settings(self, path=None, file_namespace=None):
        return self.settings_manager.load_user_settings(path,
                                                        file_namespace,
                                                        scope='experiment')

    def set_user_settings(self, key=None, value=None):
        """Set a user settings."""
        return self.settings_manager.set_user_settings(
            key, value, scope='experiment')

    def get_internal_settings(self, key):
        """Get an internal settings."""
        return self.settings_manager.get_internal_settings(key,
                                                           scope='experiment',
                                                           )

    def set_internal_settings(self, key, value):
        """Set an internal settings."""
        return self.settings_manager.set_internal_settings(key,
                                                           value,
                                                           scope='experiment',
                                                           )

    def _load_default_settings(self):
        """Load default settings for manual clustering."""
        curdir = op.dirname(op.realpath(__file__))
        # This is a namespace available in the config file.
        # file_namespace = {
        #     'n_spikes': self.model.n_spikes,
        #     'n_channels': self.model.n_channels,
        # }
        declare_namespace('manual_clustering')
        self.load_user_settings(path=op.join(curdir, 'default_settings.py'),
                                # file_namespace=file_namespace
                                )

    def _load_experiment_settings(self):
        self.settings_manager.set_experiment_path(self.experiment_path)

    # File-related actions
    # -------------------------------------------------------------------------

    def _backup_kwik(self, kwik_path):
        """Save a copy of the Kwik file before opening it."""
        backup_kwik_path = kwik_path + '.bak'
        if not op.exists(backup_kwik_path):
            info("Saving a backup of the Kwik file "
                 "in {0}.".format(backup_kwik_path))
            shutil.copyfile(kwik_path, backup_kwik_path)

    def open(self, kwik_path=None, model=None):
        """Open a .kwik file."""
        if kwik_path is not None:
            self._backup_kwik(kwik_path)
        if model is None:
            model = KwikModel(kwik_path)
        self.model = model
        self.experiment_path = (op.realpath(kwik_path)
                                if kwik_path else self.phy_user_dir)
        self.experiment_dir = op.dirname(self.experiment_path)
        self.experiment_name = model.name
        self.emit('open')

    def save(self):
        """Save the spike clusters and cluster groups to the Kwik file."""
        groups = {cluster: self.cluster_metadata.group(cluster)
                  for cluster in self.clustering.cluster_ids}
        self.model.save(self.clustering.spike_clusters,
                        groups)
        info("Saved {0:s}.".format(self.model.kwik_path))

    def close(self):
        """Close the currently-open dataset."""
        self.emit('close')
        self.model = None
        self.experiment_path = None
        self.experiment_dir = None

    # Clustering actions
    # -------------------------------------------------------------------------

    def _check_list_argument(self, arg, name='clusters'):
        if not _is_array_like(arg):
            raise ValueError("The argument should be a list or an array.")
        if len(name) == 0:
            raise ValueError("No {0} were selected.".format(name))

    def _update_selected_clusters(self, up):
        self._selected_clusters = _update_cluster_selection(
            self._selected_clusters, up)

    def select(self, clusters):
        """Select some clusters."""
        self._selected_clusters = clusters
        self.emit('select', clusters)

    def merge(self, clusters):
        """Merge some clusters."""
        up = self.clustering.merge(clusters)
        self.emit('cluster', up=up)

    def split(self, spikes):
        """Make a new cluster out of some spikes.

        Notes
        -----

        Spikes belonging to affected clusters, but not part of the `spikes`
        array, will move to brand new cluster ids. This is because a new
        cluster id must be used as soon as a cluster changes.

        """
        self._check_list_argument(spikes, 'spikes')
        up = self.clustering.split(spikes)
        self.emit('cluster', up=up)

    def move(self, clusters, group):
        """Move some clusters to a cluster group.

        Here is the list of cluster groups:

        * 0=Noise
        * 1=MUA
        * 2=Good
        * 3=Unsorted

        """
        self._check_list_argument(clusters)
        up = self.cluster_metadata.set_group(clusters, group)
        self.emit('cluster', up=up)

    def undo(self):
        """Undo the last clustering action."""
        up = self._global_history.undo()
        self.emit('cluster', up=up, add_to_stack=False)

    def redo(self):
        """Redo the last undone action."""
        up = self._global_history.redo()
        self.emit('cluster', up=up, add_to_stack=False)

    # Properties
    # -------------------------------------------------------------------------

    @property
    def cluster_ids(self):
        """Array of all cluster ids used in the current clustering."""
        return self.clustering.cluster_ids

    # Event callbacks
    # -------------------------------------------------------------------------

    def _create_cluster_metadata(self):
        self.cluster_metadata = self.model.cluster_metadata

    def _create_cluster_store(self):

        # Kwik store in experiment_dir/name.phy/1/main/cluster_store.
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

        # Initialize the progress reporter.
        pr_disk = ProgressReporter()
        pr_memory = ProgressReporter()
        self.cluster_store.register_item(FeatureMasks,
                                         progress_reporter_disk=pr_disk,
                                         progress_reporter_memory=pr_memory,
                                         )

        @pr_disk.connect
        def on_progress(value, value_max):
            if value_max == 0:
                return
            print("Initializing the cluster store: "
                  "{0:.2f}%.".format(100 * value / float(value_max)),
                  end='\r')

        @pr_memory.connect
        def on_progress(value, value_max):
            if value_max == 0:
                return
            print("Initializing cluster statistics: "
                  "{0:.2f}%.".format(100 * value / float(value_max)),
                  end='\r')

        # Generate the cluster store if it doesn't exist or is invalid.
        # If the cluster store already exists and is consistent
        # with the data, it is not recreated.
        self.cluster_store.generate(self.clustering.spikes_per_cluster)

        @self.connect
        def on_cluster(up=None, add_to_stack=None):
            self.cluster_store.on_cluster(up)

    def _create_clustering(self):
        self.clustering = Clustering(self.model.spike_clusters)

    def _create_global_history(self):
        self._global_history = GlobalHistory(process_ups=_process_ups)

        @self.connect
        def on_cluster(up=None, add_to_stack=None):
            # Update the global history.
            if add_to_stack and up is not None:
                if up.description.startswith('metadata'):
                    self._global_history.action(self.cluster_metadata)
                elif up.description in ('merge', 'assign'):
                    self._global_history.action(self.clustering)

    def _create_wizard(self):
        self.wizard = Wizard(self.clustering.cluster_ids)

        # Set the similarity and quality functions for the wizard.
        @self.wizard.set_similarity_function
        def similarity(target, candidate):
            """Compute the dot product between the mean masks of
            two clusters."""
            return np.dot(self.cluster_store.mean_masks(target),
                          self.cluster_store.mean_masks(candidate))

        @self.wizard.set_quality_function
        def quality(cluster):
            """Return the maximum mean_masks across all channels
            for a given cluster."""
            return self.cluster_store.mean_masks(cluster).max()

        @self.connect
        def on_cluster(up=None, add_to_stack=None):
            if up is None:
                return
            if up.description in ('merge', 'assign'):
                self.wizard.cluster_ids = (set(self.wizard._cluster_ids) -
                                           set(up.deleted)).union(up.added)
            elif up.description == 'metadata_group':
                if up.metadata_value in (0, 1):
                    for cluster in up.metadata_changed:
                        self.wizard.ignore(cluster)

    def on_open(self):
        """Update the session after new data has been loaded."""

        self._load_experiment_settings()

        self._create_global_history()
        self._create_clustering()
        self._create_cluster_metadata()
        self._create_cluster_store()
        self._create_wizard()

    def on_cluster(self, up=None, add_to_stack=True):
        """Update the history when clustering changes occur."""
        self._update_selected_clusters(up)
        # Update the global history.
        if add_to_stack and up is not None:
            if up.description.startswith('metadata'):
                self._global_history.action(self.cluster_metadata)
            elif up.description in ('merge', 'assign'):
                self._global_history.action(self.clustering)

    def on_close(self):
        """Save the settings when the data is closed."""
        self.settings_manager.save()

    def change_channel_group(self, channel_group):
        """Change the current channel group."""
        self.select([])
        self.model.channel_group = channel_group
        self.emit('open')

    def change_clustering(self, clustering):
        """Change the current clustering."""
        self.select([])
        self.model.clustering = clustering
        self.emit('open')

    # Wizard
    # -------------------------------------------------------------------------

    def best_clusters(self, quality=None, n_max=None):
        """Return the best clusters by decreasing order of quality.

        Parameters
        ----------

        quality : function or None
            A cluster quality function, returning a quality value for any
            cluster id. By default, the wizard's quality function is used.
        n_max : integer or None
            The maximum number of clusters to return.

        """
        if quality is None:
            return self.wizard.best_clusters(n_max=n_max)
        else:
            return _best_clusters(self.cluster_ids, quality, n_max=n_max)

    # Show views
    # -------------------------------------------------------------------------

    def _view_settings_name(self, view_name, name):
        return 'manual_clustering.{0}_{1}'.format(view_name, name)

    def _get_view_settings(self, view_name, name):
        settings_name = self._view_settings_name(view_name, name)
        return self.get_internal_settings(settings_name)

    def _set_view_settings(self, view_name, name, value):
        settings_name = self._view_settings_name(view_name, name)
        self.set_internal_settings(settings_name, value)

    def _create_view(self, view_model):
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
        def on_select(cluster_ids):
            if len(cluster_ids) == 0:
                # Clear the view.
                view.visual.empty = True
                view.update()
                return
            if view.visual.empty:
                on_open()
            view_model.on_select(cluster_ids)
            view.update()

        # Unregister the callbacks when the view is closed.
        @view.connect
        def on_close(event):
            self.unconnect(on_open, on_cluster, on_select)

            # Save the canvas position and size.
            self._set_view_settings(view_name,
                                    'canvas_position',
                                    view.position,
                                    )
            self._set_view_settings(view_name,
                                    'canvas_size',
                                    view.size,
                                    )

        # Make sure the view is correctly initialized when it is created
        # *after* that the data has been loaded.
        @view.connect
        def on_draw(event):
            if view.visual.empty:
                on_open()
                if (set(self._selected_clusters) <=
                        set(self.clustering.cluster_ids)):
                    on_select(self._selected_clusters)

        return view

    def _create_view_model(self, name, **kwargs):
        vm_class = _VIEW_MODELS[name]

        # Load the canvas position and size for that view.
        position = self._get_view_settings(name, 'canvas_position')
        size = self._get_view_settings(name, 'canvas_size')
        if position:
            kwargs['position'] = position
        if size:
            kwargs['size'] = size

        # Load the selector options for that view.
        n_spikes_max = self.get_user_settings('manual_clustering.' +
                                              name +
                                              '_n_spikes_max')
        excerpt_size = self.get_user_settings('manual_clustering.' +
                                              name +
                                              '_excerpt_size')

        return vm_class(self.model,
                        store=self.cluster_store,
                        n_spikes_max=n_spikes_max,
                        excerpt_size=excerpt_size,
                        **kwargs)

    def _create_waveforms_view(self):
        """Create a WaveformView and return a ViewModel instance."""

        # Persist scale factor.
        sf_name = 'manual_clustering.waveforms_scale_factor'
        sf = self.get_internal_settings(sf_name) or .01

        vm = self._create_view_model('waveforms',
                                     scale_factor=sf,
                                     )
        self._create_view(vm)

        # Load box scale.
        bs_name = 'manual_clustering.waveforms_box_scale'
        bs = (self.get_internal_settings(bs_name) or
              vm.view.visual.default_box_scale)
        vm.view.box_scale = bs

        # Load probe scale.
        ps_name = 'manual_clustering.waveforms_probe_scale'
        ps = (self.get_internal_settings(ps_name) or
              vm.view.visual.default_probe_scale)
        vm.view.probe_scale = ps

        @vm.view.connect
        def on_draw(event):
            # OPTIM: put this when the model or the view is closed instead
            # No need to run this at every draw!

            # Save probe and box scales.
            self.set_internal_settings(ps_name, vm.view.probe_scale)
            self.set_internal_settings(bs_name, vm.view.box_scale)

        return vm

    def _create_features_view(self):
        """Create a FeatureView and return a ViewModel instance."""

        sf_name = 'manual_clustering.features_scale_factor'
        sf = self.get_internal_settings(sf_name) or .01

        ms_name = 'manual_clustering.features_marker_size'
        ms = self.get_internal_settings(ms_name) or 2.

        vm = self._create_view_model('features',
                                     scale_factor=sf,
                                     )

        self._create_view(vm)
        vm.view.marker_size = ms

        @vm.view.connect
        def on_draw(event):
            if vm.view.visual.empty:
                return
            self.set_internal_settings(sf_name,
                                       vm.view.zoom * vm.scale_factor)
            self.set_internal_settings(ms_name,
                                       vm.view.marker_size)

        return vm

    def _create_correlograms_view(self):
        """Create a CorrelogramView and return a ViewModel instance."""
        args = 'binsize', 'winsize_bins'
        kwargs = {k: self.get_user_settings('manual_clustering.'
                                            'correlograms_' + k)
                  for k in args}
        vm = self._create_view_model('correlograms', **kwargs)
        self._create_view(vm)
        return vm

    def _create_traces_view(self):
        """Create a TraceView and return a ViewModel instance."""
        vm = self._create_view_model('traces')
        vm.scale_factor = .001
        self._create_view(vm)
        return vm

    def create_view(self, name):
        """Create a view without displaying it.

        Parameters
        ----------

        name : str
            Can be 'waveforms', 'features', or 'correlograms'.

        Returns
        -------

        view_model : ViewModel instance

        """
        if name == 'waveforms':
            return self._create_waveforms_view()
        elif name == 'features':
            return self._create_features_view()
        elif name == 'correlograms':
            return self._create_correlograms_view()
        elif name == 'traces':
            return self._create_traces_view()
        else:
            raise ValueError("The view '{0}' doesn't exist.".format(name))

    def show_waveforms(self):
        """Create and display a new Waveforms view.

        Returns
        -------

        view : VisPy canvas instance

        """
        vm = self.create_view('waveforms')
        vm.view.show()
        return vm.view

    def show_features(self):
        """Create and display a new Features view.

        Returns
        -------

        view : VisPy canvas instance

        """
        vm = self.create_view('features')
        vm.view.show()
        return vm.view

    def show_correlograms(self):
        """Create and display a new Correlograms view.

        Returns
        -------

        view : VisPy canvas instance

        """
        vm = self.create_view('correlograms')
        vm.view.show()
        return vm.view

    def show_traces(self):
        """Create and display a new Trace view.

        Returns
        -------

        view : VisPy canvas instance

        """
        vm = self.create_view('traces')
        vm.view.show()
        return vm.view
