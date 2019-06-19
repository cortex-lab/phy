# -*- coding: utf-8 -*-
# flake8: noqa

"""Plotting module based on OpenGL.

For advanced users!

"""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op

from .base import BaseVisual, GLSLInserter, BaseCanvas, BaseLayout
from .plot import PlotCanvas
from .transform import Translate, Scale, Range, Subplot, NDC, TransformChain, extend_bounds
from .panzoom import PanZoom
from .axes import AxisLocator, Axes
from .utils import get_linear_x, BatchAccumulator
from .interact import Grid, Boxed, Lasso
from .visuals import (
    ScatterVisual, UniformScatterVisual, PlotVisual, UniformPlotVisual, HistogramVisual,
    TextVisual, LineVisual, ImageVisual, PolygonVisual)
