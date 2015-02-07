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
# Session class
#------------------------------------------------------------------------------

class Session(object):
    """Provide all user-exposed actions for a manual clustering session."""

    def __init__(self, model):
        if not isinstance(model, BaseModel):
            raise ValueError("'model' must be an instance of a "
                             "class deriving from BaseModel.")
        # List of registered views.
        self._views = []
        # Dict (callback_type, view_class) => [list of callbacks]
        self._callbacks = defaultdict(list)
        # List of (create_callback, action_name) pairs.
        self._view_creators = []

        self._global_history = GlobalHistory()
        # Set the model and initialize the session.
        self.model = model
        self.update_after_load()

    @property
    def views(self):
        return self._views

    def callback(self, view_class=None):
        """Return a decorator adding a callback function."""
        def create_decorator(f):
            # on_mytype
            callback_type = f.__name__[3:]
            self._callbacks[callback_type, view_class].append(f)
            return f
        return create_decorator

    def create(self, action_name=None):
        """Callback function creating a new view."""
        def create_decorator(f):

            # Check that the decorated function name is valid.
            if hasattr(self, f.__name__):
                raise ValueError("This function name already exists in "
                                 "the Session: {0}.".format(f.__name__))

            # Wrapped function.
            @wraps(f)
            def _register_view():
                # Create the view.
                view = f()
                # Register the view.
                self._views.append(view)
                # Call all 'load' and 'select' callbacks on that view.
                # This is to make sure the view is automatically updated
                # when it is created after the data has been loaded and
                # some clusters have been selected.
                # TODO: concatenate with itertools
                for callback in self._iter_callbacks('load', view.__class__):
                    self._call_callback_on_view(callback, view)
                for callback in self._iter_callbacks('select', view.__class__):
                    self._call_callback_on_view(callback, view)

                return view

            # Assign the decorated view creator to the session.
            setattr(self, f.__name__, _register_view)

            # Register the view creators with their names.
            self._view_creators.append((_register_view, action_name))

            return _register_view
        return create_decorator

    def _iter_callbacks(self, callback_type, view_class=None):
        # Callbacks registered to a particular view class.
        if view_class is not None:
            for item in self._callbacks[callback_type, view_class]:
                yield item
        # Callbacks registered to any view class.
        for item in self._callbacks[callback_type, None]:
            yield item

    def _iter_views(self, view_class=None):
        """Iterate over all views of a certain type."""
        for view in self._views:
            for view in self._views:
                if (view_class is not None and
                   not isinstance(view, view_class)):
                    continue
                yield view

    def _call_callback_on_view(self, callback, view, **kwargs):
        """Call a callback item on a view."""
        # # Only call the callback if the view is of the correct type.
        # if (callback_item['view'] is not None and
        #    not isinstance(view, callback_item['view'])):
        #     return
        # Call the callback function on the view, with possibly an
        # 'up' instance as argument.
        if 'up' in kwargs:
            callback(view, up=kwargs['up'])
        else:
            callback(view)

    def _call_callbacks(self, callback_type, **kwargs):
        """Call all callbacks of a given type."""
        # kwargs are arguments to pass to the callbacks.
        # Loop over all views.
        for view in self._iter_views():
            # Loop over all callbacks of that type registered to that
            # view class.
            for callback in self._iter_callbacks(callback_type,
                                                 view_class=view.__class__):
                self._call_callback_on_view(callback, view, **kwargs)

    # Controller.
    # -------------------------------------------------------------------------

    def update_after_load(self):
        """Update the session after new data has been loaded."""
        # TODO: call this after the channel groups has changed.
        # Update the Selector and Clustering instances using the Model.
        spike_clusters = self.model.spike_clusters
        self.selector = Selector(spike_clusters, n_spikes_max=100)
        self.clustering = Clustering(spike_clusters)
        self.cluster_metadata = self.model.cluster_metadata
        # Update all views.
        self._call_callbacks('load')

    def update_after_select(self):
        """Update the views after the selection has changed."""
        # Update all views.
        self._call_callbacks('select')

    def update_after_cluster(self, up, add_to_stack=True):
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
        self.update_after_select()

    def merge(self, clusters):
        """Merge clusters."""
        up = self.clustering.merge(clusters)
        self.update_after_cluster(up)

    def split(self, spikes):
        """Create a new cluster from a selection of spikes."""
        up = self.clustering.split(spikes)
        self.update_after_cluster(up)

    def move(self, clusters, group):
        """Move clusters to a group."""
        self.cluster_metadata.set(clusters, 'group', group)
        self._global_history.action(self.cluster_metadata)

    def undo(self):
        """Undo the last action."""
        up = self._global_history.undo()
        self.update_after_cluster(up, add_to_stack=False)

    def redo(self):
        """Redo the last undone action."""
        up = self._global_history.redo()
        self.update_after_cluster(up, add_to_stack=False)

    def wizard_start(self):
        raise NotImplementedError()

    def wizard_next(self):
        raise NotImplementedError()

    def wizard_previous(self):
        raise NotImplementedError()

    def wizard_reset(self):
        raise NotImplementedError()
