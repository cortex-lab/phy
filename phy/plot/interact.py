# -*- coding: utf-8 -*-

"""Common layouts."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from phylib.io.array import _in_polygon
from phylib.utils.geometry import _get_boxes, _get_box_pos_size

from .base import BaseLayout
from .transform import Scale, Range, Subplot, Clip, NDC, TransformChain
from .utils import _get_texture
from .visuals import LineVisual, PolygonVisual


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
        _transforms = [
            Scale('u_grid_scaling'), Scale((ms, ms)), Clip([-mc, -mc, +mc, +mc]),
            Subplot(self.shape_var, self.box_var)]
        if has_clip is False:
            # Remove the Clip transform.
            del _transforms[2]
        self._transforms = _transforms
        self.transforms = TransformChain()
        self.transforms.add_on_gpu(_transforms, origin=self)

    def attach(self, canvas):
        """Attach the grid to a canvas."""
        super(Grid, self).attach(canvas)
        canvas.transforms += self.transforms
        canvas.inserter.insert_vert(
            """
            attribute vec2 {};
            uniform vec2 {};
            uniform vec2 u_grid_scaling;
            """.format(self.box_var, self.shape_var),
            'header', origin=self)

    def map(self, arr, box=None):
        """Apply the subplot transformation to a position array."""
        assert box is not None
        assert len(box) == self.n_dims
        arr = self._transforms[0].apply(arr)
        arr = Subplot(self.shape, box).apply(arr)
        return arr

    def imap(self, arr, box=None):
        """Apply the subplot inverse transformation to a position array."""
        assert box is not None
        arr = Subplot(self.shape, box).inverse().apply(arr)
        arr = self._transforms[0].inverse().apply(arr)
        return arr

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

    The boxes can be specified from their corner coordinates, or from their centers and
    optional sizes. If the sizes are not specified, they will be computed automatically.
    An iterative algorithm is used to find the largest box size that will not make them overlap.

    Constructor
    ----------

    box_bounds : array-like
        A (n, 4) array where each row contains the `(xmin, ymin, xmax, ymax)`
        bounds of every box, in normalized device coordinates.

        Note: the box bounds need to be contained within [-1, 1] at all times,
        otherwise an error will be raised. This is to prevent silent clipping
        of the values when they are passed to a gloo Texture2D.

    box_pos : array-like (2D, shape[1] == 2)
        Position of the centers of the boxes.
    box_size : array-like (2D, shape[1] == 2)
        Size of the boxes.

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
    _scaling = (1, 1)

    def __init__(
            self, box_bounds=None, box_pos=None, box_size=None, box_var=None,
            keep_aspect_ratio=True):
        super(Boxed, self).__init__(box_var=box_var)
        self._key_pressed = None
        self.keep_aspect_ratio = keep_aspect_ratio

        # Find the box bounds if only the box positions are passed.
        if box_bounds is None:
            assert box_pos is not None
            # This will find a good box size automatically if it is not
            # specified.
            box_bounds = _get_boxes(
                box_pos, size=box_size, keep_aspect_ratio=self.keep_aspect_ratio,
                margin=self.margin)

        self._box_bounds = np.atleast_2d(box_bounds)
        assert self._box_bounds.shape[1] == 4

        self.transforms = TransformChain()
        self.transforms.add_on_gpu([
            Scale('u_box_scaling'), Range(NDC, 'box_bounds')], origin=self)

    def attach(self, canvas):
        """Attach the boxed interact to a canvas."""
        super(Boxed, self).attach(canvas)
        canvas.transforms += self.transforms
        canvas.inserter.insert_vert("""
            #include "utils.glsl"
            attribute float {};
            uniform sampler2D u_box_bounds;
            uniform float n_boxes;
            uniform vec2 u_box_scaling;
            """.format(self.box_var), 'header', origin=self)
        canvas.inserter.insert_vert("""
            // Fetch the box bounds for the current box (`box_var`).
            vec4 box_bounds = fetch_texture({}, u_box_bounds, n_boxes);
            box_bounds = (2 * box_bounds - 1);  // See hack in Python.
            """.format(self.box_var), 'before_transforms', origin=self)

    def map(self, arr, box=None):
        """Apply the boxed transformation to a position array."""
        assert box is not None
        assert 0 <= box < len(self.box_bounds)
        return Range(NDC, self.box_bounds[box]).apply(arr)

    def imap(self, arr, box=None):
        """Apply the boxed inverse transformation to a position array."""
        assert 0 <= box < len(self.box_bounds)
        return Range(NDC, self.box_bounds[box]).inverse().apply(arr)

    def update_visual(self, visual):
        """Update a visual."""
        super(Boxed, self).update_visual(visual)
        # Signal bounds (positions).
        box_bounds = _get_texture(self._box_bounds, NDC, self.n_boxes, [-1, 1])
        box_bounds = box_bounds.astype(np.float32)
        if 'u_box_bounds' in visual.program:
            visual.program['u_box_bounds'] = box_bounds
            visual.program['n_boxes'] = self.n_boxes
            visual.program['u_box_scaling'] = self._scaling

    def add_boxes(self, canvas):
        """Show the boxes borders."""
        n_boxes = len(self.box_bounds)
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
        return self._box_bounds

    @box_bounds.setter
    def box_bounds(self, val):
        self._box_bounds = np.atleast_2d(val)
        assert self._box_bounds.shape[1] == 4
        self.update()

    @property
    def box_pos(self):
        """Position of the box centers."""
        box_pos, _ = _get_box_pos_size(self._box_bounds)
        return box_pos

    @box_pos.setter
    def box_pos(self, val):
        self.box_bounds = _get_boxes(
            val, size=self.box_size, margin=self.margin,
            keep_aspect_ratio=self.keep_aspect_ratio)

    @property
    def box_size(self):
        """Sizes of the boxes."""
        _, box_size = _get_box_pos_size(self._box_bounds)
        return box_size

    @box_size.setter
    def box_size(self, val):
        assert len(val) == 2
        self.box_bounds = _get_boxes(
            self.box_pos, size=val, margin=self.margin,
            keep_aspect_ratio=self.keep_aspect_ratio)

    def get_closest_box(self, pos):
        """Get the box closest to some position."""
        pos = np.atleast_2d(pos)
        x0, y0, x1, y1 = self.box_bounds.T
        rmin = np.c_[x0, y0]
        rmax = np.c_[x1, y1]
        z = np.zeros_like(rmin)
        d = np.maximum(np.maximum(rmin - pos, z), pos - rmax)
        return np.argmin(np.linalg.norm(d, axis=1))

    def update_boxes(self, box_pos, box_size):
        """Set the box bounds from specified box positions and sizes."""
        assert box_pos.shape == (self.n_boxes, 2)
        assert len(box_size) == 2
        self.box_bounds = _get_boxes(
            box_pos, size=box_size, margin=self.margin,
            keep_aspect_ratio=self.keep_aspect_ratio)

    # Scaling
    #--------------------------------------------------------------------------

    @property
    def scaling(self):
        """Return the grid scaling."""
        return self._scaling

    @scaling.setter
    def scaling(self, value):
        self._scaling = value
        self.update()


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

    def __init__(self, n_boxes, box_var=None, origin=None):
        self.origin = origin
        b = self.get_box_bounds(n_boxes)
        super(Stacked, self).__init__(b, box_var=box_var, keep_aspect_ratio=False)

    @property
    def n_boxes(self):
        """Number of boxes."""
        return len(self.box_pos)

    @n_boxes.setter
    def n_boxes(self, n_boxes):
        if n_boxes >= 1:
            self.box_bounds = self.get_box_bounds(n_boxes)

    def get_box_bounds(self, n_boxes):
        """Return the box bounds for a given number of stacked boxes."""
        # The margin must be in [-1, 1]
        margin = .05
        margin = np.clip(margin, -1, 1)
        # Normalize the margin.
        margin = 2. * margin / float(n_boxes)

        # Signal bounds.
        b = np.zeros((n_boxes, 4))
        b[:, 0] = -1
        b[:, 1] = np.linspace(-1, 1 - 2. / n_boxes + margin, n_boxes)
        b[:, 2] = 1
        b[:, 3] = np.linspace(-1 + 2. / n_boxes - margin, 1., n_boxes)
        origin = self.origin or 'top'
        if origin == 'top':
            b = b[::-1, :]
        return b


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
        if not self.visual:
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
            else:
                self.clear()
                self.box = None

    def __repr__(self):
        return str(self.polygon)
