# -*- coding: utf-8 -*-

"""Session structure."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from collections import defaultdict
from functools import wraps

import numpy as np

from ._history import GlobalHistory
from .clustering import Clustering
from .cluster_view import ClusterView
from .cluster_metadata import ClusterMetadata
from .selector import Selector
from ...io.base_model import BaseModel
from ...plot.waveforms import WaveformView
from ...notebook.utils import load_css


#------------------------------------------------------------------------------
# View manager
#------------------------------------------------------------------------------

class CallbackManager(object):
    """Manage the callbacks for the different views."""
    def __init__(self, session):
        self._session = session
        self._callbacks = defaultdict(list)
        # List of (create_callback, action_name) pairs.
        self._view_creators = []

    def callbacks(self, *callback_types):
        """Return all callbacks registered for a given callback type."""
        # List of callback functions, for all specified callback types.
        l = [self._callbacks[callback_type]
             for callback_type in callback_types]
        # Flatten the list of lists.
        return [item for sublist in l for item in sublist]

    def _call_callback_on_view(self, callback_item, view, **kwargs):
        """Call a callback item on a view."""
        # Only call the callback if the view is of the correct type.
        if not isinstance(view, callback_item['view']):
            return
        # Call the callback function on the view, with possibly an
        # 'up' instance as argument.
        if 'up' in kwargs:
            callback_item['callback'](view, up=kwargs['up'])
        else:
            callback_item['callback'](view)

    def _decorator(self, callback_type, **kwargs):
        """Return a decorator adding a callback function."""
        def decorator(f):
            item = {'callback': f}
            item.update(kwargs)
            self._callbacks[callback_type].append(item)
        return decorator

    def create(self, action_name=None):
        """Callback function creating a new view."""
        def create_decorator(f):

            # Check that the decorated function name is valid.
            if hasattr(self._session, f.__name__):
                raise ValueError("This function name already exists in "
                                 "the Session: {0}.".format(f.__name__))

            # Wrapped function.
            @wraps(f)
            def _register_view():
                # Create the view.
                view = f()
                # Register the view.
                self._session._views.append(view)
                # Call all 'load' and 'select' callbacks on that view.
                # This is to make sure the view is automatically updated
                # when it is created after the data has been loaded and
                # some clusters have been selected.
                for callback in self.callbacks('load', 'select'):
                    self._call_callback_on_view(callback, view)

            # Assign the decorated view creator to the session.
            setattr(self._session, f.__name__, _register_view)

            # Register the view creators with their names.
            self._view_creators.append((_register_view, action_name))

            return _register_view
        return create_decorator

    def load(self, view=None):
        """Callback function when a dataset is loaded."""
        return self._decorator('load', view=view)

    def select(self, view=None):
        """Callback function when clusters are selected."""
        return self._decorator('select', view=view)

    def cluster(self, callback, view=None):
        """Callback function when the clustering changes."""
        return self._decorator('cluster', view=view)


#------------------------------------------------------------------------------
# Session class
#------------------------------------------------------------------------------

class Session(object):
    """Provide all user-exposed actions for a manual clustering session."""

    def __init__(self, model):
        if not isinstance(model, BaseModel):
            raise ValueError("'model' must be an instance of a "
                             "class deriving from BaseModel.")
        self._global_history = GlobalHistory()
        self._callback_manager = CallbackManager(self)
        self._views = []
        # Set the model and initialize the session.
        self.model = model
        self._update_after_load()

    @property
    def views(self):
        return self._views

    def _iter_views(self, view_class=None):
        """Iterate over all views of a certain type."""
        for view in self._views:
            for view in self._views:
                if (view_class is not None and
                   not isinstance(view, view_class)):
                    continue
                yield view

    def _call_callbacks(self, callback_type, **kwargs):
        """Call all callbacks of a given type."""
        # kwargs are arguments to pass to the callbacks.
        for item in self._callback_manager.callbacks(callback_type):
            # 'item' is a dictionary containing information about the callback.
            # item['callback'] is the callback function itself.
            assert 'view' in item
            for view in self._iter_views(item['view']):
                self._callback_manager._call_callback_on_view(item,
                                                              view,
                                                              **kwargs)

    # Controller.
    # -------------------------------------------------------------------------

    def _update_after_load(self):
        """Update the session after new data has been loaded."""
        # Update the Selector and Clustering instances using the Model.
        spike_clusters = self.model.spike_clusters
        self.selector = Selector(spike_clusters, n_spikes_max=100)
        self.clustering = Clustering(spike_clusters)
        self.cluster_metadata = self.model.cluster_metadata
        # Update all views.
        self._call_callbacks('load')

    def _update_after_select(self):
        """Update the views after the selection has changed."""
        # Update all views.
        self._call_callbacks('select')

    def _update_after_cluster(self, up, add_to_stack=True):
        """Update the session after the clustering has changed."""

        # TODO: Update the similarity matrix.
        # stats.update(up)

        # TODO: this doesn't do anything yet.
        # self.selector.update(up)

        # TODO: this doesn't do anything yet.
        # self.cluster_metadata.update(up)

        if add_to_stack:
            self._global_history.action(self.clustering)

        # Update all views.
        self._call_callbacks('cluster', up=up)

    # Public properties.
    # -------------------------------------------------------------------------

    @property
    def cluster_labels(self):
        """Labels of all current clusters."""
        return self.clustering.cluster_labels

    @property
    def cluster_colors(self):
        """Colors of all current clusters."""
        return [self.cluster_metadata[cluster]['color']
                for cluster in self.clustering.cluster_labels]

    # Public clustering actions.
    # -------------------------------------------------------------------------

    def select(self, clusters):
        """Select some clusters."""
        self.selector.selected_clusters = clusters
        self._update_after_select()

    def merge(self, clusters):
        """Merge clusters."""
        up = self.clustering.merge(clusters)
        self._update_after_cluster(up)

    def split(self, spikes):
        """Create a new cluster from a selection of spikes."""
        up = self.clustering.split(spikes)
        self._update_after_cluster(up)

    def move(self, clusters, group):
        """Move clusters to a group."""
        self.cluster_metadata.set(clusters, 'group', group)
        self._global_history.action(self.cluster_metadata)

    def undo(self):
        """Undo the last action."""
        up = self._global_history.undo()
        self._update_after_cluster(up, add_to_stack=False)

    def redo(self):
        """Redo the last undone action."""
        up = self._global_history.redo()
        self._update_after_cluster(up, add_to_stack=False)

    def wizard_start(self):
        raise NotImplementedError()

    def wizard_next(self):
        raise NotImplementedError()

    def wizard_previous(self):
        raise NotImplementedError()

    def wizard_reset(self):
        raise NotImplementedError()
