# -*- coding: utf-8 -*-

"""UpdatedData class."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np


#------------------------------------------------------------------------------
# UpdatedData class
#------------------------------------------------------------------------------

class UpdatedData(object):
    description = None  # optional information about the update
    clusters = None  # all clusters affected by the update
    spikes = None  # all spikes affected by the update
    added_clusters = None  # new clusters
    deleted_clusters = None  # deleted clusters
    changed_clusters = None  # clusters with updated metadata
