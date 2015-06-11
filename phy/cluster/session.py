# -*- coding: utf-8 -*-
from __future__ import print_function

"""Session structure."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op
import shutil

from ..utils.logging import info, warn
from ..utils.settings import _ensure_dir_exists
from ..io.base_model import BaseSession
from ..io.kwik.model import KwikModel
from ..io.kwik.store_items import create_store
from .manual.gui import ClusterManualGUI
from .launcher import KlustaKwik


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


class Session(BaseSession):
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

    _vm_classes = ClusterManualGUI._vm_classes
    _gui_classes = {'cluster_manual': ClusterManualGUI}

    def __init__(self,
                 kwik_path=None,
                 clustering=None,
                 model=None,
                 use_store=True,
                 phy_user_dir=None,
                 ):
        self._clustering = clustering
        self._use_store = use_store
        curdir = op.dirname(op.realpath(__file__))
        settings_path = op.join(curdir, 'default_settings.py')
        if kwik_path:
            kwik_path = op.realpath(kwik_path)
        super(Session, self).__init__(model=model,
                                      path=kwik_path,
                                      phy_user_dir=phy_user_dir,
                                      default_settings_path=settings_path,
                                      vm_classes=self._vm_classes,
                                      gui_classes=self._gui_classes,
                                      )

    def _pre_open(self):
        @self.connect
        def on_open():
            self.settings['on_open'](self)

    def _backup_kwik(self, kwik_path):
        """Save a copy of the Kwik file before opening it."""
        if kwik_path is None:
            return
        backup_kwik_path = kwik_path + '.bak'
        if not op.exists(backup_kwik_path):
            info("Saving a backup of the Kwik file "
                 "in {0}.".format(backup_kwik_path))
            shutil.copyfile(kwik_path, backup_kwik_path)

    def _create_model(self, path):
        model = KwikModel(path, clustering=self._clustering)
        return model

    def _save_model(self):
        """Save the spike clusters and cluster groups to the Kwik file."""
        groups = {cluster: self.model.cluster_metadata.group(cluster)
                  for cluster in self.cluster_ids}
        self.model.save(self.model.spike_clusters,
                        groups,
                        clustering_metadata=self.model.clustering_metadata,
                        )
        info("Saved {0:s}.".format(self.model.kwik_path))

    # File-related actions
    # -------------------------------------------------------------------------

    def open(self, kwik_path=None, model=None):
        self._backup_kwik(kwik_path)
        return super(Session, self).open(model=model, path=kwik_path)

    @property
    def kwik_path(self):
        return self.model.path

    @property
    def has_unsaved_changes(self):
        """Whether there are unsaved changes in the model.

        If true, a prompt message for saving will be displayed when closing
        the GUI.

        """
        # TODO
        pass

    # Properties
    # -------------------------------------------------------------------------

    @property
    def cluster_ids(self):
        """Array of all cluster ids used in the current clustering."""
        return self.model.cluster_ids

    @property
    def n_clusters(self):
        """Number of clusters in the current clustering."""
        return self.model.n_clusters

    # Event callbacks
    # -------------------------------------------------------------------------

    def _create_cluster_store(self):
        if not self._use_store:
            # Just use a mock store.
            self.store = create_store(self.model,
                                      self.model.spikes_per_cluster,
                                      )
            return

        # Kwik store in experiment_dir/name.phy/1/main/cluster_store.
        store_path = op.join(self.settings.exp_settings_dir,
                             'cluster_store',
                             str(self.model.channel_group),
                             self.model.clustering
                             )
        _ensure_dir_exists(store_path)

        # Instantiate the store.
        spc = self.model.spikes_per_cluster
        cs = self.settings['features_masks_chunk_size']
        wns = self.settings['waveforms_n_spikes_max']
        wes = self.settings['waveforms_excerpt_size']
        self.store = create_store(self.model,
                                  path=store_path,
                                  spikes_per_cluster=spc,
                                  features_masks_chunk_size=cs,
                                  waveforms_n_spikes_max=wns,
                                  waveforms_excerpt_size=wes,
                                  )

        # Generate the cluster store if it doesn't exist or is invalid.
        # If the cluster store already exists and is consistent
        # with the data, it is not recreated.
        self.store.generate()

    def change_channel_group(self, channel_group):
        """Change the current channel group."""
        self.model.channel_group = channel_group
        info("Switched to channel group {}.".format(channel_group))
        self.emit('open')

    def change_clustering(self, clustering):
        """Change the current clustering."""
        self._clustering = clustering
        self.model.clustering = clustering
        info("Switched to `{}` clustering.".format(clustering))
        self.emit('open')

    def on_open(self):
        self._create_cluster_store()

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
                self.store.memory_store.store(cluster, **{name: out})

            # Add the statistics.
            stats = self.store.items['statistics']
            stats.add(name, _wrapper, shape)
            # Register it in the global cluster store.
            self.store.register_field(name, 'statistics')
            # Compute it on all existing clusters.
            stats.store_all(name=name, mode='force')
            info("Registered statistic `{}`.".format(name))

        return decorator

    # Automatic clustering
    # -------------------------------------------------------------------------

    def cluster(self,
                clustering=None,
                algorithm='klustakwik',
                spike_ids=None,
                **kwargs):
        """Run an automatic clustering algorithm on all or some of the spikes.

        Parameters
        ----------

        clustering : str
            The name of the clustering in which to save the results.
        algorithm : str
            The algorithm name. Only `klustakwik` currently.
        spike_ids : array-like
            Array of spikes to cluster.

        Returns
        -------

        spike_clusters : array
            The spike_clusters assignements returned by the algorithm.

        """
        if clustering is None:
            clustering = 'original'
        # Make sure the clustering name does not exist already.
        if clustering in self.model.clusterings:
            old = clustering
            i = 0
            while True:
                new = '{}_{}'.format(clustering, i)
                if new not in self.model.clusterings:
                    break
                i += 1
            clustering = new
            warn("The clustering `{}` already exists -- ".format(old) +
                 "switching to `{}`.".format(new))
        kk = KlustaKwik(**kwargs)
        info("Running {}...".format(algorithm))
        # Run KK.
        sc = kk.cluster(model=self.model, spike_ids=spike_ids)
        # Save the results in the Kwik file.
        spike_clusters = self.model.spike_clusters.copy()
        spike_clusters[spike_ids] = sc
        # Add a new clustering and switch to it.
        self.model.add_clustering(clustering, spike_clusters)
        self.change_clustering(clustering)
        # Set the new clustering metadata.
        params = kk.params
        params['version'] = kk.version
        metadata = {'{}_{}'.format(algorithm, name): value
                    for name, value in params.items()}
        self.model.clustering_metadata.update(metadata)
        self.save()
        info("The automatic clustering has finished.")
        info("The clustering has been saved in the "
             "`{}` clustering in the `.kwik` file.".format(clustering))
        return sc

    # GUI
    # -------------------------------------------------------------------------

    def show_gui(self, **kwargs):
        gui = super(Session, self).show_gui(store=self.store,
                                            **kwargs)

        @gui.connect
        def on_request_save():
            self.save()

        return gui
