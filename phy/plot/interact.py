# -*- coding: utf-8 -*-

"""Common layouts."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import numpy as np

from phylib.utils import emit
from phylib.utils.geometry import get_non_overlapping_boxes, get_closest_box

from .base import BaseLayout
from .transform import Scale, Range, Subplot, Clip, NDC
from .utils import _get_texture, _in_polygon
from .visuals import LineVisual, PolygonVisual

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Grid
#------------------------------------------------------------------------------

class Grid(BaseLayout):
    """Layout showing subplots arranged in a 2D grid.

    Constructor
    -----------

    shape : tuple or str
        Number of rows, cols in the grid.
    shape_var : str
        Name of the GLSL uniform variable that holds the shape, when it is variable.
    box_var : str
        Name of the GLSL variable with the box index.
    has_clip : boolean
        Whether subplots should be clipped.

    Note
    ----

    To be used in a grid, a visual must define `a_box_index` (by default) or another GLSL
    variable specified in `box_var`.

    """

    margin = .075
    n_dims = 2
    active_box = (0, 0)
    _scaling = (1., 1.)

    def __init__(self, shape=(1, 1), shape_var='u_grid_shape', box_var=None, has_clip=True):
        super(Grid, self).__init__(box_var=box_var)
        self.shape_var = shape_var
        self._shape = shape
        ms = 1 - self.margin
        mc = 1 - self.margin

        # Define the GPU transforms of the Grid layout.
        # 1. Global scaling.
        self.gpu_transforms.add(Scale(self._scaling, gpu_var='u_grid_scaling'))
        # 2. Margin.
        self.gpu_transforms.add(Scale((ms, ms)))
        # 3. Clipping for the subplots.
        if has_clip:
            self.gpu_transforms.add(Clip([-mc, -mc, +mc, +mc]))
        # 4. Subplots.
        self.gpu_transforms.add(Subplot(
            # The parameters of the subplots are callable as they can be changed dynamically.
            shape=lambda: self._shape, index=lambda: self.active_box,
            shape_gpu_var=self.shape_var, index_gpu_var=self.box_var))

    def attach(self, canvas):
        """Attach the grid to a canvas."""
        super(Grid, self).attach(canvas)
        canvas.gpu_transforms += self.gpu_transforms
        canvas.inserter.insert_vert(
            """
            attribute vec2 {};
            uniform vec2 {};
            uniform vec2 u_grid_scaling;
            """.format(self.box_var, self.shape_var),
            'header', origin=self)

    def add_boxes(self, canvas, shape=None):
        """Show subplot boxes."""
        shape = shape or self.shape
        assert isinstance(shape, tuple)
        n, m = shape
        n_boxes = n * m
        a = 1 - .0001

        pos = np.array([[-a, -a, +a, -a],
                        [+a, -a, +a, +a],
                        [+a, +a, -a, +a],
                        [-a, +a, -a, -a],
                        ])
        pos = np.tile(pos, (n_boxes, 1))

        box_index = []
        for i in range(n):
            for j in range(m):
                box_index.append([i, j])
        box_index = np.vstack(box_index)
        box_index = np.repeat(box_index, 8, axis=0)

        boxes = LineVisual()

        # We exclude this interact when adding the visual.
        canvas.add_visual(boxes, clearable=False)
        boxes.set_data(pos=pos)
        boxes.set_box_index(box_index)
        canvas.update()

    def get_closest_box(self, pos):
        """Get the box index (i, j) closest to a given position in NDC coordinates."""
        x, y = pos
        rows, cols = self.shape
        j = np.clip(int(cols * (1. + x) / 2.), 0, cols - 1)
        i = np.clip(int(rows * (1. - y) / 2.), 0, rows - 1)
        return i, j

    def update_visual(self, visual):
        """Update a visual."""
        super(Grid, self).update_visual(visual)
        if self.shape_var in visual.program:
            visual.program[self.shape_var] = self._shape
            visual.program['u_grid_scaling'] = self._scaling

    @property
    def shape(self):
        """Return the grid shape."""
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = value
        self.update()

    @property
    def scaling(self):
        """Return the grid scaling."""
        return self._scaling

    @scaling.setter
    def scaling(self, value):
        self._scaling = value
        self.update()


#------------------------------------------------------------------------------
# Boxed
#------------------------------------------------------------------------------

class Boxed(BaseLayout):
    """Layout showing plots in rectangles at arbitrary positions. Used by the waveform view.

    The boxes are specified via their center positions and optional sizes, in which case
    an iterative algorithm is used to find the largest box size that will not make them overlap.

    Constructor
    ----------

    box_pos : array-like (2D, shape[1] == 2)
        Position of the centers of the boxes.
    box_var : str
        Name of the GLSL variable with the box index.
    keep_aspect_ratio : boolean
        Whether to keep the aspect ratio of the bounds.

    Note
    ----

    To be used in a boxed layout, a visual must define `a_box_index` (by default) or another GLSL
    variable specified in `box_var`.

    """

    margin = .1
    n_dims = 1
    active_box = 0
    _box_scaling = (1., 1.)
    _layout_scaling = (1., 1.)
    _scaling_param_increment = 1.1

    def __init__(self, box_pos=None, box_var=None, keep_aspect_ratio=False):
        super(Boxed, self).__init__(box_var=box_var)
        self._key_pressed = None
        self.keep_aspect_ratio = keep_aspect_ratio

        self.update_boxes(box_pos)

        self.gpu_transforms.add(Range(
            NDC, lambda: self.box_bounds[self.active_box],
            from_gpu_var='vec4(-1, -1, 1, 1)', to_gpu_var='box_bounds'))

    def attach(self, canvas):
        """Attach the boxed interact to a canvas."""
        super(Boxed, self).attach(canvas)
        canvas.gpu_transforms += self.gpu_transforms
        canvas.inserter.insert_vert("""
            #include "utils.glsl"
            attribute float {};
            uniform sampler2D u_box_pos;
            uniform float n_boxes;
            uniform vec2 u_box_size;
            uniform vec2 u_layout_scaling;
            """.format(self.box_var), 'header', origin=self)
        canvas.inserter.insert_vert("""
            // Fetch the box bounds for the current box (`box_var`).
            vec2 box_pos = fetch_texture({}, u_box_pos, n_boxes).xy;
            box_pos = (2 * box_pos - 1);  // from [0, 1] (texture) to [-1, 1] (NDC)
            box_pos = box_pos * u_layout_scaling;
            vec4 box_bounds = vec4(box_pos - u_box_size, box_pos + u_box_size);
            """.format(self.box_var), 'start', origin=self)

    def update_visual(self, visual):
        """Update a visual."""
        super(Boxed, self).update_visual(visual)
        box_pos = _get_texture(self.box_pos, (0, 0), self.n_boxes, [-1, 1])
        box_pos = box_pos.astype(np.float32)
        if 'u_box_pos' in visual.program:
            logger.log(5, "Update visual with interact Boxed.")
            visual.program['u_box_pos'] = box_pos
            visual.program['n_boxes'] = self.n_boxes
            visual.program['u_box_size'] = np.array(self.box_size) * np.array(self._box_scaling)
            visual.program['u_layout_scaling'] = self._layout_scaling

    def update_boxes(self, box_pos):
        """Update the box positions and automatically-computed size."""
        self.box_pos, self.box_size = get_non_overlapping_boxes(box_pos)

    def add_boxes(self, canvas):
        """Show the boxes borders."""
        n_boxes = len(self.box_pos)
        a = 1 + .05

        pos = np.array([[-a, -a, +a, -a],
                        [+a, -a, +a, +a],
                        [+a, +a, -a, +a],
                        [-a, +a, -a, -a],
                        ])
        pos = np.tile(pos, (n_boxes, 1))

        boxes = LineVisual()
        box_index = np.repeat(np.arange(n_boxes), 8)

        canvas.add_visual(boxes, clearable=False)
        boxes.set_data(pos=pos, color=(.5, .5, .5, 1))
        boxes.set_box_index(box_index)
        canvas.update()

    # Change the box bounds, positions, or size
    #--------------------------------------------------------------------------

    @property
    def n_boxes(self):
        """Total number of boxes."""
        return len(self.box_pos)

    @property
    def box_bounds(self):
        """Bounds of the boxes."""
        bs = np.array(self.box_size)
        return np.c_[self.box_pos - bs, self.box_pos + bs]

    def get_closest_box(self, pos):
        """Get the box closest to some position."""
        return get_closest_box(pos, self.box_pos, self.box_size)

    # Box scaling
    #--------------------------------------------------------------------------

    def _increment_box_scaling(self, cw=1., ch=1.):
        self._box_scaling = (self._box_scaling[0] * cw, self._box_scaling[1] * ch)
        self.update()

    @property
    def box_scaling(self):
        return self._box_scaling

    def expand_box_width(self):
        return self._increment_box_scaling(cw=self._scaling_param_increment)

    def shrink_box_width(self):
        return self._increment_box_scaling(cw=1. / self._scaling_param_increment)

    def expand_box_height(self):
        return self._increment_box_scaling(ch=self._scaling_param_increment)

    def shrink_box_height(self):
        return self._increment_box_scaling(ch=1. / self._scaling_param_increment)

    # Layout scaling
    #--------------------------------------------------------------------------

    def _increment_layout_scaling(self, cw=1., ch=1.):
        self._layout_scaling = (self._layout_scaling[0] * cw, self._layout_scaling[1] * ch)
        self.update()

    @property
    def layout_scaling(self):
        return self._layout_scaling

    def expand_layout_width(self):
        return self._increment_layout_scaling(cw=self._scaling_param_increment)

    def shrink_layout_width(self):
        return self._increment_layout_scaling(cw=1. / self._scaling_param_increment)

    def expand_layout_height(self):
        return self._increment_layout_scaling(ch=self._scaling_param_increment)

    def shrink_layout_height(self):
        return self._increment_layout_scaling(ch=1. / self._scaling_param_increment)


class Stacked(Boxed):
    """Layout showing a number of subplots stacked vertically.

    Parameters
    ----------

    n_boxes : int
        Number of boxes to stack vertically.
    box_var : str
        Name of the GLSL variable with the box index.
    origin : str
        top or bottom

    Note
    ----

    To be used in a boxed layout, a visual must define `a_box_index` (by default) or another GLSL
    variable specified in `box_var`.

    """
    margin = 0
    _origin = 'bottom'

    def __init__(self, n_boxes, box_var=None, origin=None):
        self._origin = origin or self._origin
        assert self._origin in ('top', 'bottom')
        box_pos = self.get_box_pos(n_boxes)
        super(Stacked, self).__init__(box_pos, box_var=box_var, keep_aspect_ratio=False)

    @property
    def n_boxes(self):
        """Number of boxes."""
        return len(self.box_pos)

    @n_boxes.setter
    def n_boxes(self, n_boxes):
        if n_boxes >= 1:
            self.update_boxes(self.get_box_pos(n_boxes))

    def get_box_pos(self, n_boxes):
        """Return the box bounds for a given number of stacked boxes."""
        # Signal bounds.
        b = np.zeros((n_boxes, 2))
        b[:, 1] = np.linspace(-1, 1, n_boxes)
        if self._origin == 'top':
            b = b[::-1, :]
        return b

    @property
    def origin(self):
        """Whether to show the channels from top to bottom (`top` option, the default), or from
        bottom to top (`bottom`)."""
        return self._origin

    @origin.setter
    def origin(self, value):
        self._origin = value
        self.update_boxes(self.get_box_pos(self.n_boxes))
        self.update()

    def attach(self, canvas):
        """Attach the stacked interact to a canvas."""
        BaseLayout.attach(self, canvas)
        canvas.gpu_transforms += self.gpu_transforms
        canvas.inserter.insert_vert("""
            #include "utils.glsl"
            attribute float {};
            uniform float n_boxes;
            uniform bool u_top_origin;
            uniform vec2 u_box_size;
            """.format(self.box_var), 'header', origin=self)
        canvas.inserter.insert_vert("""
            float margin = .1 / n_boxes;
            float a = 1 - 2. / n_boxes + margin;
            float b = -1 + 2. / n_boxes - margin;
            float u = (u_top_origin ? (n_boxes - 1. - {bv}) : {bv}) / max(1., n_boxes - 1.);
            float y0 = -1 + u * (a + 1);
            float y1 = b + u * (1 - b);
            float ym = .5 * (y0 + y1);
            float yh = u_box_size.y * (y1 - ym);
            y0 = ym - yh;
            y1 = ym + yh;
            vec4 box_bounds = vec4(-1., y0, +1., y1);
        """.format(bv=self.box_var), 'before_transforms', origin=self)

    def update_visual(self, visual):
        """Update a visual."""
        BaseLayout.update_visual(self, visual)
        if 'n_boxes' in visual.program:
            visual.program['n_boxes'] = self.n_boxes
            visual.program['u_box_size'] = self._box_scaling
            visual.program['u_top_origin'] = self._origin == 'top'


#------------------------------------------------------------------------------
# Interactive tools
#------------------------------------------------------------------------------

class Lasso(object):
    """Draw a polygon with the mouse and find the points that belong to the inside of the
    polygon."""
    def __init__(self):
        self._points = []
        self.canvas = None
        self.visual = None
        self.box = None

    def add(self, pos):
        """Add a point to the polygon."""
        x, y = pos.flat if isinstance(pos, np.ndarray) else pos
        self._points.append((x, y))
        logger.debug("Lasso has %d points.", len(self._points))
        self.update_lasso_visual()

    @property
    def polygon(self):
        """Coordinates of the polygon vertices."""
        l = self._points
        # Close the polygon.
        # l = l + l[0] if len(l) else l
        out = np.array(l, dtype=np.float64)
        out = np.reshape(out, (out.size // 2, 2))
        assert out.ndim == 2
        assert out.shape[1] == 2
        return out

    def clear(self):
        """Reset the lasso."""
        self._points = []
        self.box = None
        self.update_lasso_visual()

    @property
    def count(self):
        """Number of vertices in the polygon."""
        return len(self._points)

    def in_polygon(self, pos):
        """Return which points belong to the polygon."""
        return _in_polygon(pos, self.polygon)

    def attach(self, canvas):
        """Attach the lasso to a canvas."""
        canvas.attach_events(self)
        self.canvas = canvas
        self.create_lasso_visual()

    def create_lasso_visual(self):
        """Create the lasso visual."""
        self.visual = PolygonVisual()
        self.canvas.add_visual(self.visual, clearable=False)

    def update_lasso_visual(self):
        """Update the lasso visual with the current polygon."""
        if not self.visual and self.count > 0:
            return
        # The following call updates a_box_index with the active box in BaseLayout.
        self.visual.set_data(pos=self.polygon)
        self.canvas.update()

    def on_mouse_click(self, e):
        """Add a polygon point with ctrl+click."""
        if 'Control' in e.modifiers:
            if e.button == 'Left':
                layout = getattr(self.canvas, 'layout', None)
                if hasattr(layout, 'box_map'):
                    box, pos = layout.box_map(e.pos)
                    # Only update the box for the first click, so that the box containing
                    # the lasso is determined by the first click only.
                    if self.box is None:
                        self.box = box
                    # Avoid clicks outside the active box (box of the first click).
                    if box != self.box:
                        return
                else:  # pragma: no cover
                    pos = self.canvas.window_to_ndc(e.pos)
                # Force the active box to be the box of the first click, not the box of the
                # current click.
                if layout:
                    layout.active_box = self.box
                self.add(pos)  # call update_lasso_visual
                emit("lasso_updated", self.canvas, self.polygon)
            else:
                self.clear()
                self.box = None

    def __repr__(self):
        return str(self.polygon)
