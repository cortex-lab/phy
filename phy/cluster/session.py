# -*- coding: utf-8 -*-
from __future__ import print_function

"""Session structure."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op
import shutil

from ..utils.logging import info
from ..utils.settings import _ensure_dir_exists
from ..io.kwik.model import KwikModel
from ..io.kwik.store_items import create_store
from .manual.gui import ClusterManualGUI
from ..gui.base import BaseSession


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

    def __init__(self, kwik_path=None, model=None, phy_user_dir=None):
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
        model = KwikModel(path)
        return model

    def _save_model(self):
        """Save the spike clusters and cluster groups to the Kwik file."""
        groups = {cluster: self.model.cluster_metadata.group(cluster)
                  for cluster in self.cluster_ids}
        self.model.save(self.model.spike_clusters, groups)
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
        self.emit('open')

    def change_clustering(self, clustering):
        """Change the current clustering."""
        self.model.clustering = clustering
        self.emit('open')

    def on_open(self):
        self._create_cluster_store()

    # Views and GUIs
    # -------------------------------------------------------------------------

    def show_gui(self, **kwargs):
        return super(Session, self).show_gui(store=self.store,
                                             **kwargs)
