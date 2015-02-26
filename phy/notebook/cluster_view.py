# -*- coding: utf-8 -*-

"""Cluster view in HTML."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from IPython.utils.traitlets import Unicode, List
from IPython.html.widgets import DOMWidget


#------------------------------------------------------------------------------
# Cluster view
#------------------------------------------------------------------------------

class ClusterView(DOMWidget):
    _view_name = Unicode('ClusterWidget', sync=True)
    description = Unicode(help="Description", sync=True)
    clusters = List(sync=True)
    colors = List(sync=True)
    value = List(sync=True)
