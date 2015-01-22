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


def update_info(**kwargs):
    """Hold information about clustering changes."""
    d = dict(
        description=None,  # optional information about the update
        spikes=[],  # all spikes affected by the update
        added=[],  # new clusters
        deleted=[],  # deleted clusters
        count_changed=[],  # clusters with changed count
        metadata_changed=[]  # clusters with changed metadata
    )
    d.update(kwargs)
    return AttrDict(d)


UpdateInfo = update_info
