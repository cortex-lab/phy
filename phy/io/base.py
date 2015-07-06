# -*- coding: utf-8 -*-

"""The BaseModel class holds the data from an experiment."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op
from collections import defaultdict

import numpy as np

from ..ext import six
from ..utils import debug, EventEmitter
from ..utils._types import _as_list, _is_list
from ..utils.settings import (Settings,
                              _ensure_dir_exists,
                              _phy_user_dir,
                              )
from ..gui.base import WidgetCreator


#------------------------------------------------------------------------------
# ClusterMetadata class
#------------------------------------------------------------------------------

class ClusterMetadata(object):
    """Hold cluster metadata.

    Features
    --------

    * New metadata fields can be easily registered
    * Arbitrary functions can be used for default values

    Notes
    ----

    If a metadata field `group` is registered, then two methods are
    dynamically created:

    * `group(cluster)` returns the group of a cluster, or the default value
      if the cluster doesn't exist.
    * `set_group(cluster, value)` sets a value for the `group` metadata field.

    """
    def __init__(self, data=None):
        self._fields = {}
        self._data = defaultdict(dict)
        # Fill the existing values.
        if data is not None:
            self._data.update(data)

    @property
    def data(self):
        return self._data

    def _get_one(self, cluster, field):
        """Return the field value for a cluster, or the default value if it
        doesn't exist."""
        if cluster in self._data:
            if field in self._data[cluster]:
                return self._data[cluster][field]
            elif field in self._fields:
                # Call the default field function.
                return self._fields[field](cluster)
            else:
                return None
        else:
            if field in self._fields:
                return self._fields[field](cluster)
            else:
                return None

    def _get(self, clusters, field):
        if _is_list(clusters):
            return [self._get_one(cluster, field)
                    for cluster in _as_list(clusters)]
        else:
            return self._get_one(clusters, field)

    def _set_one(self, cluster, field, value):
        """Set a field value for a cluster."""
        self._data[cluster][field] = value

    def _set(self, clusters, field, value):
        clusters = _as_list(clusters)
        for cluster in clusters:
            self._set_one(cluster, field, value)

    def default(self, func):
        """Register a new metadata field with a function
        returning the default value of a cluster."""
        field = func.__name__
        # Register the decorated function as the default field function.
        self._fields[field] = func
        # Create self.<field>(clusters).
        setattr(self, field, lambda clusters: self._get(clusters, field))
        # Create self.set_<field>(clusters, value).
        setattr(self, 'set_{0:s}'.format(field),
                lambda clusters, value: self._set(clusters, field, value))
        return func


#------------------------------------------------------------------------------
# BaseModel class
#------------------------------------------------------------------------------

class BaseModel(object):
    """This class holds data from an experiment.

    This base class must be derived.

    """
    def __init__(self):
        self.name = 'model'
        self._channel_group = None
        self._clustering = None

    @property
    def path(self):
        return None

    # Channel groups
    # -------------------------------------------------------------------------

    @property
    def channel_group(self):
        return self._channel_group

    @channel_group.setter
    def channel_group(self, value):
        assert isinstance(value, six.integer_types)
        self._channel_group = value
        self._channel_group_changed(value)

    def _channel_group_changed(self, value):
        """Called when the channel group changes.

        May be implemented by child classes.

        """
        pass

    @property
    def channel_groups(self):
        """List of channel groups.

        May be implemented by child classes.

        """
        return []

    # Clusterings
    # -------------------------------------------------------------------------

    @property
    def clustering(self):
        return self._clustering

    @clustering.setter
    def clustering(self, value):
        # The clustering is specified by a string.
        assert isinstance(value, six.string_types)
        self._clustering = value
        self._clustering_changed(value)

    def _clustering_changed(self, value):
        """Called when the clustering changes.

        May be implemented by child classes.

        """
        pass

    @property
    def clusterings(self):
        """List of clusterings.

        May be implemented by child classes.

        """
        return []

    # Data
    # -------------------------------------------------------------------------

    @property
    def metadata(self):
        """A dictionary holding metadata about the experiment.

        May be implemented by child classes.

        """
        raise NotImplementedError()

    @property
    def traces(self):
        """Traces (may be memory-mapped).

        May be implemented by child classes.

        """
        raise NotImplementedError()

    @property
    def spike_samples(self):
        """Spike times from the current channel_group.

        Must be implemented by child classes.

        """
        raise NotImplementedError()

    @property
    def sample_rate(self):
        pass

    @property
    def spike_times(self):
        """Spike times from the current channel_group.

        This is a NumPy array containing `float64` values (in seconds).

        The spike times of all recordings are concatenated. There is no gap
        between consecutive recordings, currently.

        """
        return self.spike_samples.astype(np.float64) / self.sample_rate

    @property
    def spike_clusters(self):
        """Spike clusters from the current channel_group.

        Must be implemented by child classes.

        """
        raise NotImplementedError()

    def spike_train(self, cluster_id):
        """Return the spike times of a given cluster."""
        return self.spike_times[self.spikes_per_cluster[cluster_id]]

    @property
    def spikes_per_cluster(self):
        """Spikes per cluster dictionary.

        Must be implemented by child classes.

        """
        raise NotImplementedError()

    def update_spikes_per_cluster(self, spc):
        raise NotImplementedError()

    @property
    def cluster_metadata(self):
        """ClusterMetadata instance holding information about the clusters.

        Must be implemented by child classes.

        """
        raise NotImplementedError()

    @property
    def cluster_groups(self):
        """Groups of all clusters in the current channel group and clustering.

        This is a regular Python dictionary.

        """
        return {cluster: self.cluster_metadata.group(cluster)
                for cluster in self.cluster_ids}

    @property
    def features(self):
        """Features from the current channel_group (may be memory-mapped).

        May be implemented by child classes.

        """
        raise NotImplementedError()

    @property
    def masks(self):
        """Masks from the current channel_group (may be memory-mapped).

        May be implemented by child classes.

        """
        raise NotImplementedError()

    @property
    def waveforms(self):
        """Waveforms from the current channel_group (may be memory-mapped).

        May be implemented by child classes.

        """
        raise NotImplementedError()

    @property
    def probe(self):
        """A Probe instance.

        May be implemented by child classes.

        """
        raise NotImplementedError()

    def save(self):
        """Save the data.

        May be implemented by child classes.

        """
        raise NotImplementedError()

    def close(self):
        """Close the model and the underlying files.

        May be implemented by child classes.

        """
        pass


#------------------------------------------------------------------------------
# Session
#------------------------------------------------------------------------------

class BaseSession(EventEmitter):
    """Give access to the data, views, and GUIs in an interactive session.

    The model must implement:

    * `model(path)`
    * `model.path`
    * `model.close()`

    Events
    ------

    open
    close

    """
    def __init__(self,
                 model=None,
                 path=None,
                 phy_user_dir=None,
                 default_settings_path=None,
                 vm_classes=None,
                 gui_classes=None,
                 ):
        super(BaseSession, self).__init__()

        self.model = None
        if phy_user_dir is None:
            phy_user_dir = _phy_user_dir()
        _ensure_dir_exists(phy_user_dir)
        self.phy_user_dir = phy_user_dir

        self._create_settings(default_settings_path)

        if gui_classes is None:
            gui_classes = self.settings['gui_classes']
        self._gui_creator = WidgetCreator(widget_classes=gui_classes)

        self.connect(self.on_open)
        self.connect(self.on_close)

        # Custom `on_open()` callback function.
        if 'on_open' in self.settings:
            @self.connect
            def on_open():
                self.settings['on_open'](self)

        self._pre_open()
        if model or path:
            self.open(path, model=model)

    def _create_settings(self, default_settings_path):
        self.settings = Settings(phy_user_dir=self.phy_user_dir,
                                 default_path=default_settings_path,
                                 )

        @self.connect
        def on_open():
            # Initialize the settings with the model's path.
            self.settings.on_open(self.experiment_path)

    # Methods to override
    # -------------------------------------------------------------------------

    def _pre_open(self):
        pass

    def _create_model(self, path):
        """Create a model from a path.

        Must be overriden.

        """
        pass

    def _save_model(self):
        """Save a model.

        Must be overriden.

        """
        pass

    def on_open(self):
        pass

    def on_close(self):
        pass

    # File-related actions
    # -------------------------------------------------------------------------

    def open(self, path=None, model=None):
        """Open a dataset."""
        # Close the session if it is already open.
        if self.model:
            self.close()
        if model is None:
            model = self._create_model(path)
        self.model = model
        self.experiment_path = (op.realpath(path)
                                if path else self.phy_user_dir)
        self.emit('open')

    def reopen(self):
        self.open(model=self.model)

    def save(self):
        self._save_model()

    def close(self):
        """Close the currently-open dataset."""
        self.model.close()
        self.emit('close')
        self.model = None

    # Views and GUIs
    # -------------------------------------------------------------------------

    def show_gui(self, name=None, show=True, **kwargs):
        """Show a new GUI."""
        if name is None:
            gui_classes = list(self._gui_creator.widget_classes.keys())
            if gui_classes:
                name = gui_classes[0]

        # Get the default GUI config.
        params = {p: self.settings.get('{}_{}'.format(name, p), None)
                  for p in ('config', 'shortcuts', 'snippets', 'state')}
        params.update(kwargs)

        # Create the GUI.
        gui = self._gui_creator.add(name,
                                    model=self.model,
                                    settings=self.settings,
                                    **params)
        gui._save_state = True

        # Connect the 'open' event.
        self.connect(gui.on_open)

        @gui.main_window.connect_
        def on_close_gui():
            self.unconnect(gui.on_open)
            # Save the params of every view in the GUI.
            for vm in gui.views:
                self.save_view_params(vm, save_size_pos=False)
            gs = gui.main_window.save_geometry_state()
            gs['view_count'] = gui.view_count()
            if not gui._save_state:
                gs['state'] = None
                gs['geometry'] = None
            self.settings['{}_state'.format(name)] = gs
            self.settings.save()

        # HACK: do not save GUI state when views have been closed or reset
        # in the session, otherwise Qt messes things up in the GUI.
        @gui.connect
        def on_close_view(view):
            gui._save_state = False

        @gui.connect
        def on_reset_gui():
            gui._save_state = False

        # Custom `on_gui_open()` callback.
        if 'on_gui_open' in self.settings:
            self.settings['on_gui_open'](self, gui)

        if show:
            gui.show()

        return gui

    def save_view_params(self, vm, save_size_pos=True):
        """Save the parameters exported by a view model instance."""
        to_save = vm.exported_params(save_size_pos=save_size_pos)
        for key, value in to_save.items():
            assert vm.name
            name = '{}_{}'.format(vm.name, key)
            self.settings[name] = value
            debug("Save {0}={1} for {2}.".format(name, value, vm.name))
