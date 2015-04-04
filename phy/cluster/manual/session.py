# -*- coding: utf-8 -*-

"""Session structure."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os
import os.path as op
from functools import partial
import shutil

import numpy as np

from ...ext.six import string_types
from ...utils._misc import (_phy_user_dir,
                            _ensure_phy_user_dir_exists)
from ...ext.slugify import slugify
from ...utils.event import EventEmitter
from ...utils.logging import set_level, warn
from ...io.kwik_model import KwikModel
from ...io.base_model import BaseModel
from ._history import GlobalHistory
from ._utils import _concatenate_per_cluster_arrays
from .cluster_info import ClusterMetadata
from .clustering import Clustering
from .selector import Selector
from .store import ClusterStore, StoreItem


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
    fields = [('masks', 'disk'),
              ('mean_masks', 'memory')]

    def store_from_model(self, cluster, spikes):
        # Only load the masks from the model if the masks aren't already
        # stored.
        to_store = {}

        masks = self.store.load(cluster, 'masks')

        if masks is None or masks.shape[0] != len(spikes):
            # Load all features and masks for that cluster in memory.
            masks = self.model.masks[spikes]
            to_store.update(masks=masks)

        to_store.update(mean_masks=masks.mean(axis=0))
        self.store.store(cluster, **to_store)


class Waveforms(StoreItem):
    fields = [('spikes', 'memory'),
              ('waveforms', 'custom')]

    def store_from_model(self, cluster, spikes):
        # Save the spikes in the cluster.
        self.store.store(cluster, spikes=spikes)

    def load(self, cluster, spikes=None):
        if spikes is None:
            spikes = self.store.load(cluster, 'spikes')
            assert spikes is not None
        return self.model.waveforms[spikes]


#------------------------------------------------------------------------------
# Session class
#------------------------------------------------------------------------------

def _ensure_disk_store_exists(dir_name, root_path=None):
    # Disk store.
    if root_path is None:
        _ensure_phy_user_dir_exists()
        root_path = _phy_user_dir('cluster_store')
        # Create the disk store if it does not exist.
        if not op.exists(root_path):
            os.mkdir(root_path)
    if not op.exists(root_path):
        raise RuntimeError("Please create the store directory "
                           "{0}".format(root_path))
    # Put the store in a subfolder, using the name.
    dir_name = slugify(dir_name)
    path = op.join(root_path, dir_name)
    if not op.exists(path):
        os.mkdir(path)
    return path


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
    """Default manual clustering session.

    Parameters
    ----------
    filename : str
        Path to a .kwik file, to be used if 'model' is not used.
    model : instance of BaseModel
        A Model instance, to be used if 'filename' is not used.

    """
    def __init__(self, store_path=None):
        super(Session, self).__init__()
        self.model = None
        self._store_path = store_path

        # self.action and self.connect are decorators.
        self.action(self.open, title='Open')
        self.action(self.select, title='Select clusters')
        self.action(self.merge, title='Merge')
        self.action(self.split, title='Split')
        self.action(self.move, title='Move clusters to a group')
        self.action(self.undo, title='Undo')
        self.action(self.redo, title='Redo')

        self.connect(self.on_open)
        self.connect(self.on_cluster)

    # Public actions
    # -------------------------------------------------------------------------

    def open(self, filename=None, model=None):
        if model is None:
            model = KwikModel(filename)
        self.model = model
        self.emit('open')

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

    # Event callbacks
    # -------------------------------------------------------------------------

    def on_open(self):
        """Update the session after new data has been loaded."""
        self._global_history = GlobalHistory(process_ups=_process_ups)
        # TODO: call this after the channel groups has changed.
        # Update the Selector and Clustering instances using the Model.
        spike_clusters = self.model.spike_clusters
        self.clustering = Clustering(spike_clusters)
        self.cluster_metadata = self.model.cluster_metadata
        # TODO: n_spikes_max in a user parameter
        self.selector = Selector(spike_clusters, n_spikes_max=100)

        # Kwik store.
        path = _ensure_disk_store_exists(self.model.name,
                                         root_path=self._store_path)
        self.store = ClusterStore(model=self.model, path=path)
        self.store.register_item(FeatureMasks)
        self.store.register_item(Waveforms)
        # TODO: do not reinitialize the store every time the dataset
        # is loaded! Check if the store exists and check consistency.
        self.store.generate(self.clustering.spikes_per_cluster)

        @self.connect
        def on_cluster(up=None, add_to_stack=None):
            self.store.update(up)

    def on_cluster(self, up=None, add_to_stack=True):
        if add_to_stack:
            self._global_history.action(self.clustering)
            # TODO: if metadata
            # self._global_history.action(self.cluster_metadata)
