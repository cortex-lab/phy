# -*- coding: utf-8 -*-

"""UpdateInfo class."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np


#------------------------------------------------------------------------------
# UpdateInfo class
#------------------------------------------------------------------------------

class AttrDict(dict):
    """Like a dict, but also supports __getitem__ in addition to
    __getattr__."""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class UpdateInfo(AttrDict):
    """Hold information about clustering changes."""
    def __init___(self, **kwargs):
        # Default dictionary.
        d = dict(
            description=None,  # optional information about the update
            spikes=None,  # all spikes affected by the update
            added_clusters=None,  # new clusters
            deleted_clusters=None,  # deleted clusters
            changed_clusters=None  # clusters with updated metadata
        )
        # Update with provided keyword arguments.
        d.update(kwargs)
        self.__init__(d)
