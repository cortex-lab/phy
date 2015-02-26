# -*- coding: utf-8 -*-

"""UpdateInfo class."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ...utils._bunch import Bunch


#------------------------------------------------------------------------------
# UpdateInfo class
#------------------------------------------------------------------------------

def update_info(**kwargs):
    """Hold information about clustering changes."""
    d = dict(
        description=None,  # optional information about the update
        spikes=[],  # all spikes affected by the update
        added=[],  # new clusters
        deleted=[],  # deleted clusters
        descendants=[],  # pairs of (old_cluster, new_cluster)
        metadata_changed=[]  # clusters with changed metadata
    )
    d.update(kwargs)
    return Bunch(d)


UpdateInfo = update_info
