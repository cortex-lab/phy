# -*- coding: utf-8 -*-

"""The BaseModel class holds the data from an experiment."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from ..ext import six
from collections import defaultdict

from ..utils._types import _as_list, _is_list


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
    def spike_clusters(self):
        """Spike clusters from the current channel_group.

        Must be implemented by child classes.

        """
        raise NotImplementedError()

    @property
    def cluster_metadata(self):
        """ClusterMetadata instance holding information about the clusters.

        Must be implemented by child classes.

        """
        raise NotImplementedError()

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
