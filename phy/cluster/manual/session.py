# -*- coding: utf-8 -*-
from __future__ import print_function

"""Session structure."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------


import os.path as op
import shutil

import numpy as np

from ...utils.event import EventEmitter
from ...utils.logging import info
from ...utils.settings import (Settings,
                               _ensure_dir_exists,
                               _phy_user_dir,
                               )
from ...io.kwik.model import KwikModel, cluster_group_id
from ...io.kwik.store_items import create_store
from ._history import GlobalHistory
from ._utils import ClusterMetadataUpdater
from .clustering import Clustering
from .views import ViewCreator
from .gui import GUICreator
from .wizard import Wizard


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


class Session(EventEmitter):
    """A manual clustering session.

    This is the main object used for manual clustering. It implements
    all common actions:

    * Loading a dataset (`.kwik` file)
    * Listing the clusters
    * Changing the current channel group or current clustering
    * Showing views (waveforms, features, correlograms, etc.)
    * Clustering actions: merge, split, undo, redo
    * Wizard: cluster quality, best clusters, most similar clusters
    * Save back to .kwik

    """
    def __init__(self, kwik_path=None, phy_user_dir=None):
        super(Session, self).__init__()
        self.model = None
        if phy_user_dir is None:
            phy_user_dir = _phy_user_dir()
        _ensure_dir_exists(phy_user_dir)
        self.phy_user_dir = phy_user_dir
        # True if there are possibly unsaved changes.
        self._is_dirty = False
        self._create_settings()
        self._create_creators()

        self.connect(self.on_open)
        self.connect(self.on_close)

        if kwik_path:
            self.open(kwik_path)

    def _create_creators(self):
        self.view_creator = ViewCreator(self)
        self.gui_creator = GUICreator(self)

    def _create_settings(self):
        curdir = op.dirname(op.realpath(__file__))
        self.settings = Settings(phy_user_dir=self.phy_user_dir,
                                 default_path=op.join(curdir,
                                                      'default_settings.py'))

        @self.connect
        def on_open():
            self.settings.on_open(self.experiment_path)

    # File-related actions
    # -------------------------------------------------------------------------

    @property
    def has_unsaved_changes(self):
        """Whether there are unsaved changes in the model.

        If true, a prompt message for saving will be displayed when closing
        the GUI.

        """
        return self._is_dirty

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
        # Close the session if it is already open.
        if self.model:
            self.close()
        self.model = model
        self.experiment_path = (op.realpath(kwik_path)
                                if kwik_path else self.phy_user_dir)
        self.emit('open')

    def save(self):
        """Save the spike clusters and cluster groups to the Kwik file."""
        groups = {cluster: self._cluster_metadata_updater.group(cluster)
                  for cluster in self.clustering.cluster_ids}
        self.model.save(self.clustering.spike_clusters,
                        groups)
        info("Saved {0:s}.".format(self.model.kwik_path))
        self._is_dirty = False

    def close(self):
        """Close the currently-open dataset."""
        self.emit('close')
        self.model = None
        self.experiment_path = None

    # Clustering actions
    # -------------------------------------------------------------------------

    def _check_list_argument(self, arg, name='clusters'):
        if not isinstance(arg, (list, tuple, np.ndarray)):
            raise ValueError("The argument should be a list or an array.")
        if len(name) == 0:
            raise ValueError("No {0} were selected.".format(name))

    def merge(self, clusters):
        """Merge some clusters."""
        clusters = list(clusters)
        if len(clusters) <= 1:
            return
        self._is_dirty = True
        up = self.clustering.merge(clusters)
        info("Merge clusters {} to {}.".format(str(clusters),
                                               str(up.added[0])))
        self._global_history.action(self.clustering)
        self.emit('cluster', up=up)
        return up

    def split(self, spikes):
        """Make a new cluster out of some spikes.

        Notes
        -----

        Spikes belonging to affected clusters, but not part of the `spikes`
        array, will move to brand new cluster ids. This is because a new
        cluster id must be used as soon as a cluster changes.

        """
        self._check_list_argument(spikes, 'spikes')
        info("Split {0:d} spikes.".format(len(spikes)))
        self._is_dirty = True
        up = self.clustering.split(spikes)
        self._global_history.action(self.clustering)
        self.emit('cluster', up=up)
        return up

    def move(self, clusters, group):
        """Move some clusters to a cluster group.

        Here is the list of cluster groups:

        * 0=Noise
        * 1=MUA
        * 2=Good
        * 3=Unsorted

        """
        self._check_list_argument(clusters)
        info("Move clusters {0} to {1}.".format(str(clusters), group))
        self._is_dirty = True
        group_id = cluster_group_id(group)
        up = self._cluster_metadata_updater.set_group(clusters, group_id)
        self._global_history.action(self._cluster_metadata_updater)
        # Extra UpdateInfo fields.
        # up.update(kwargs)
        self.emit('cluster', up=up)
        return up

    def _undo_redo(self, up):
        if up:
            info("{} {}.".format(up.history.title(),
                                 up.description,
                                 ))
            self.emit('cluster', up=up)

    def undo(self):
        """Undo the last clustering action."""
        up = self._global_history.undo()
        self._undo_redo(up)
        return up

    def redo(self):
        """Redo the last undone action."""
        up = self._global_history.redo()
        self._undo_redo(up)
        return up

    # Properties
    # -------------------------------------------------------------------------

    @property
    def cluster_ids(self):
        """Array of all cluster ids used in the current clustering."""
        return self.clustering.cluster_ids

    @property
    def n_clusters(self):
        """Number of clusters in the current clustering."""
        return self.clustering.n_clusters

    # Customization methods
    # -------------------------------------------------------------------------

    def register_statistic(self, func=None, shape=(-1,)):
        """Decorator registering a custom cluster statistic.

        Parameters
        ----------

        func : function
            A function that takes a cluster index as argument, and returns
            some statistics (generally a NumPy array).

        Notes
        -----

        This function will be called on every cluster when a dataset is opened.
        It is also automatically called on new clusters when clusters change.
        You can access the data from the model and from the cluster store.

        """
        if func is not None:
            return self.register_statistic()(func)

        def decorator(func):

            name = func.__name__

            def _wrapper(cluster):
                out = func(cluster)
                self.cluster_store.memory_store.store(cluster, **{name: out})

            # Add the statistics.
            stats = self.cluster_store.items['statistics']
            stats.add(name, _wrapper, shape)
            # Register it in the global cluster store.
            self.cluster_store.register_field(name, 'statistics')
            # Compute it on all existing clusters.
            stats.store_all(name=name, mode='force')
            info("Registered statistic `{}`.".format(name))

        return decorator

    # Event callbacks
    # -------------------------------------------------------------------------

    def _create_cluster_store(self):

        # Kwik store in experiment_dir/name.phy/1/main/cluster_store.
        store_path = op.join(self.settings.exp_settings_dir,
                             'cluster_store',
                             str(self.model.channel_group),
                             self.model.clustering
                             )
        _ensure_dir_exists(store_path)

        # Instantiate the store.
        spc = self.clustering.spikes_per_cluster
        cs = self.settings['features_masks_chunk_size']
        wns = self.settings['waveforms_n_spikes_max']
        wes = self.settings['waveforms_excerpt_size']
        self.cluster_store = create_store(self.model,
                                          path=store_path,
                                          spikes_per_cluster=spc,
                                          features_masks_chunk_size=cs,
                                          waveforms_n_spikes_max=wns,
                                          waveforms_excerpt_size=wes,
                                          )

        # Generate the cluster store if it doesn't exist or is invalid.
        # If the cluster store already exists and is consistent
        # with the data, it is not recreated.
        self.cluster_store.generate()

        @self.connect
        def on_cluster(up=None):
            # No need to delete the old clusters from the store, we can keep
            # them for possible undo, and regularly clean up the store.
            for item in self.cluster_store.items.values():
                item.on_cluster(up)

    def _create_cluster_metadata(self):
        self._cluster_metadata_updater = ClusterMetadataUpdater(
            self.model.cluster_metadata)

    def _create_clustering(self):
        self.clustering = Clustering(self.model.spike_clusters)

    def _create_global_history(self):
        self._global_history = GlobalHistory(process_ups=_process_ups)

    def _to_wizard_group(self, group_id):
        """Return the group name required by the wizard, as a function
        of the Kwik cluster group."""
        if hasattr(group_id, '__len__'):
            group_id = group_id[0]
        return {
            0: 'ignored',
            1: 'ignored',
            2: 'good',
            3: None,
            None: None,
        }.get(group_id, 'good')

    def _create_wizard(self):

        # Initialize the groups for the wizard.
        def _group(cluster):
            group_id = self._cluster_metadata_updater.group(cluster)
            return self._to_wizard_group(group_id)

        groups = {cluster: _group(cluster)
                  for cluster in self.clustering.cluster_ids}
        self.wizard = Wizard(groups)

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
        def on_cluster(up):
            # HACK: get the current group as it is not available in `up`
            # currently.
            if up.description.startswith('metadata'):
                up = up.copy()
                cluster = up.metadata_changed[0]
                group = self.model.cluster_metadata.group(cluster)
                up.metadata_value = self._to_wizard_group(group)
            # This called for both regular and history actions.
            # Save the wizard selection and update the wizard.
            self.wizard.on_cluster(up)

    def on_open(self):
        """Update the session after new data has been loaded."""
        self._create_global_history()
        self._create_clustering()
        self._create_cluster_metadata()
        self._create_cluster_store()
        self._create_wizard()

    def on_close(self):
        """Save the settings when the data is closed."""
        self.settings.save()

    def change_channel_group(self, channel_group):
        """Change the current channel group."""
        self.model.channel_group = channel_group
        self.emit('open')

    def change_clustering(self, clustering):
        """Change the current clustering."""
        self.model.clustering = clustering
        self.emit('open')

    # Views and GUIs
    # -------------------------------------------------------------------------

    def show_gui(self, config=None, **kwargs):
        """Show a new manual clustering GUI."""
        # Ensure that a Qt application is running.
        gui = self.gui_creator.add(config, **kwargs)
        return gui

    def show_view(self, name, cluster_ids, **kwargs):
        """Create and display a new view.

        Parameters
        ----------

        name : str
            Can be `waveforms`, `features`, `correlograms`, or `traces`.
        cluster_ids : array-like
            List of clusters to show.

        Returns
        -------

        vm : `ViewModel` instance

        """
        do_show = kwargs.pop('show', True)
        vm = self.view_creator.add(name,
                                   show=do_show,
                                   cluster_ids=cluster_ids,
                                   **kwargs)
        if do_show:
            vm.view.show()
        return vm
