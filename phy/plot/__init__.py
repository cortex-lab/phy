# -*- coding: utf-8 -*-
# flake8: noqa

"""VisPy plotting."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op

from vispy import config

# from .interact import Grid, Stacked, Boxed
# from .plot import GridView, BoxedView, StackedView  # noqa
from .transform import Translate, Scale, Range, Subplot, NDC
from .panzoom import PanZoom
from.visuals import _get_linear_x


#------------------------------------------------------------------------------
# Add the `glsl/ path` for shader include
#------------------------------------------------------------------------------

curdir = op.dirname(op.realpath(__file__))
glsl_path = op.join(curdir, 'glsl')
if not config['include_path']:
    config['include_path'] = [glsl_path]
