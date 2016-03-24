# -*- coding: utf-8 -*-

"""Common interacts."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from vispy.gloo import Texture2D

from .base import BaseInteract
from .transform import Scale, Range, Subplot, Clip, NDC
from .utils import _get_texture, _get_boxes, _get_box_pos_size
from .visuals import LineVisual


#------------------------------------------------------------------------------
# Grid interact
#------------------------------------------------------------------------------

class Grid(BaseInteract):
    """Grid interact.

    NOTE: to be used in a grid, a visual must define `a_box_index`
    (by default) or another GLSL variable specified in `box_var`.

    Parameters
    ----------

    shape : tuple or str
        Number of rows, cols in the grid.
    box_var : str
        Name of the GLSL variable with the box index.

    """

    margin = .075

    def __init__(self, shape=(1, 1), shape_var='u_grid_shape', box_var=None):
        # Name of the variable with the box index.
        self.box_var = box_var or 'a_box_index'
        self.shape_var = shape_var
        self._shape = shape
        ms = 1 - self.margin
        mc = 1 - self.margin
        self._transforms = [Scale((ms, ms)),
                            Clip([-mc, -mc, +mc, +mc]),
                            Subplot(self.shape_var, self.box_var),
                            ]

    def attach(self, canvas):
        super(Grid, self).attach(canvas)
        canvas.transforms.add_on_gpu(self._transforms)
        canvas.inserter.insert_vert("""
                                    attribute vec2 {};
                                    uniform vec2 {};
                                    """.format(self.box_var, self.shape_var),
                                    'header')

    def map(self, arr, box=None):
        assert box is not None
        assert len(box) == 2
        arr = self._transforms[0].apply(arr)
        arr = Subplot(self.shape, box).apply(arr)
        return arr

    def imap(self, arr, box=None):
        assert box is not None
        arr = Subplot(self.shape, box).inverse().apply(arr)
        arr = self._transforms[0].inverse().apply(arr)
        return arr

    def add_boxes(self, canvas, shape=None):
        shape = shape or self.shape
        assert isinstance(shape, tuple)
        n, m = shape
        n_boxes = n * m
        a = 1 + .05

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

        @boxes.set_canvas_transforms_filter
        def _remove_clip(tc):
            return tc.remove('Clip')

        canvas.add_visual(boxes)
        boxes.set_data(pos=pos)
        boxes.program['a_box_index'] = box_index.astype(np.float32)

    def get_closest_box(self, pos):
        x, y = pos
        rows, cols = self.shape
        j = np.clip(int(cols * (1. + x) / 2.), 0, cols - 1)
        i = np.clip(int(rows * (1. - y) / 2.), 0, rows - 1)
        return i, j

    def update_program(self, program):
        program[self.shape_var] = self._shape
        # Only set the default box index if necessary.
        try:
            program[self.box_var]
        except KeyError:
            program[self.box_var] = (0, 0)

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = value
        self.update()


#------------------------------------------------------------------------------
# Boxed interact
#------------------------------------------------------------------------------

class Boxed(BaseInteract):
    """Boxed interact.

    NOTE: to be used in a boxed, a visual must define `a_box_index`
    (by default) or another GLSL variable specified in `box_var`.

    Parameters
    ----------

    box_bounds : array-like
        A (n, 4) array where each row contains the `(xmin, ymin, xmax, ymax)`
        bounds of every box, in normalized device coordinates.

        NOTE: the box bounds need to be contained within [-1, 1] at all times,
        otherwise an error will be raised. This is to prevent silent clipping
        of the values when they are passed to a gloo Texture2D.

    box_var : str
        Name of the GLSL variable with the box index.

    """
    def __init__(self,
                 box_bounds=None,
                 box_pos=None,
                 box_size=None,
                 box_var=None,
                 keep_aspect_ratio=True,
                 ):
        self._key_pressed = None
        self.keep_aspect_ratio = keep_aspect_ratio

        # Name of the variable with the box index.
        self.box_var = box_var or 'a_box_index'

        # Find the box bounds if only the box positions are passed.
        if box_bounds is None:
            assert box_pos is not None
            # This will find a good box size automatically if it is not
            # specified.
            box_bounds = _get_boxes(box_pos, size=box_size,
                                    keep_aspect_ratio=self.keep_aspect_ratio)

        self._box_bounds = np.atleast_2d(box_bounds)
        assert self._box_bounds.shape[1] == 4

        self.n_boxes = len(self._box_bounds)
        self._transforms = [Range(NDC, 'box_bounds')]

    def attach(self, canvas):
        super(Boxed, self).attach(canvas)
        canvas.transforms.add_on_gpu(self._transforms)
        canvas.inserter.insert_vert("""
            #include "utils.glsl"
            attribute float {};
            uniform sampler2D u_box_bounds;
            uniform float n_boxes;""".format(self.box_var), 'header')
        canvas.inserter.insert_vert("""
            // Fetch the box bounds for the current box (`box_var`).
            vec4 box_bounds = fetch_texture({},
                                            u_box_bounds,
                                            n_boxes);
            box_bounds = (2 * box_bounds - 1);  // See hack in Python.
            """.format(self.box_var), 'before_transforms')

    def map(self, arr, box=None):
        assert box is not None
        assert 0 <= box < len(self.box_bounds)
        return Range(NDC, self.box_bounds[box]).apply(arr)

    def imap(self, arr, box=None):
        assert 0 <= box < len(self.box_bounds)
        return Range(NDC, self.box_bounds[box]).inverse().apply(arr)

    def update_program(self, program):
        # Signal bounds (positions).
        box_bounds = _get_texture(self._box_bounds, NDC, self.n_boxes, [-1, 1])
        box_bounds = box_bounds.astype(np.float32)
        # TODO OPTIM: set the texture at initialization and update the data
        program['u_box_bounds'] = Texture2D(box_bounds,
                                            internalformat='rgba32f')
        program['n_boxes'] = self.n_boxes

    # Change the box bounds, positions, or size
    #--------------------------------------------------------------------------

    @property
    def box_bounds(self):
        return self._box_bounds

    @box_bounds.setter
    def box_bounds(self, val):
        assert val.shape == (self.n_boxes, 4)
        self._box_bounds = val
        self.update()

    @property
    def box_pos(self):
        box_pos, _ = _get_box_pos_size(self._box_bounds)
        return box_pos

    @box_pos.setter
    def box_pos(self, val):
        assert val.shape == (self.n_boxes, 2)
        self.box_bounds = _get_boxes(val, size=self.box_size,
                                     keep_aspect_ratio=self.keep_aspect_ratio)

    @property
    def box_size(self):
        _, box_size = _get_box_pos_size(self._box_bounds)
        return box_size

    @box_size.setter
    def box_size(self, val):
        assert len(val) == 2
        self.box_bounds = _get_boxes(self.box_pos, size=val,
                                     keep_aspect_ratio=self.keep_aspect_ratio)

    def get_closest_box(self, pos):
        """Get the box closest to some position."""
        pos = np.atleast_2d(pos)
        d = np.sum((np.array(self.box_pos) - pos) ** 2, axis=1)
        idx = np.argmin(d)
        return idx

    def update_boxes(self, box_pos, box_size):
        """Set the box bounds from specified box positions and sizes."""
        assert box_pos.shape == (self.n_boxes, 2)
        assert len(box_size) == 2
        self.box_bounds = _get_boxes(box_pos,
                                     size=box_size,
                                     keep_aspect_ratio=self.keep_aspect_ratio,
                                     )


class Stacked(Boxed):
    """Stacked interact.

    NOTE: to be used in a stacked, a visual must define `a_box_index`
    (by default) or another GLSL variable specified in `box_var`.

    Parameters
    ----------

    n_boxes : int
        Number of boxes to stack vertically.
    margin : int (0 by default)
        The margin between the stacked subplots. Can be negative. Must be
        between -1 and 1. The unit is relative to each box's size.
    box_var : str
        Name of the GLSL variable with the box index.

    """
    def __init__(self, n_boxes, margin=0, box_var=None, origin=None):

        # The margin must be in [-1, 1]
        margin = np.clip(margin, -1, 1)
        # Normalize the margin.
        margin = 2. * margin / float(n_boxes)

        # Signal bounds.
        b = np.zeros((n_boxes, 4))
        b[:, 0] = -1
        b[:, 1] = np.linspace(-1, 1 - 2. / n_boxes + margin, n_boxes)
        b[:, 2] = 1
        b[:, 3] = np.linspace(-1 + 2. / n_boxes - margin, 1., n_boxes)
        if origin == 'upper':
            b = b[::-1, :]

        super(Stacked, self).__init__(b, box_var=box_var,
                                      keep_aspect_ratio=False,
                                      )
