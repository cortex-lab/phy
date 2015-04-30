# -*- coding: utf-8 -*-

"""Pan & zoom transform."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import math

import numpy as np

# from vispy import gloo

from ..utils.array import _as_array


#------------------------------------------------------------------------------
# PanZoom class
#------------------------------------------------------------------------------

class PanZoom(object):
    """Pan & zoom transform.

    The panzoom transform allow to translate and scale an object in the window
    space coordinate (2D). This means that whatever point you grab on the
    screen, it should remains under the mouse pointer. Zoom is realized using
    the mouse scroll and is always centered on the mouse pointer.

    You can also control programmatically the transform using:

    * aspect: control the aspect ratio of the whole scene
    * pan   : translate the scene to the given 2D coordinates
    * zoom  : set the zoom level (centered at current pan coordinates)
    * zmin  : minimum zoom level
    * zmax  : maximum zoom level

    Interactivity
    -------------

    Pan:

    * Mouse : left-click and move (drag movement)
    * Keyboard : arrows

    Zoom:

    * Mouse : wheel or right-click and move (drag movement)
    * Keyboard : + and -

    Reset:

    * Keyboard : R

    """

    _default_zoom_coeff = 1.5
    _default_wheel_coeff = .1

    def __init__(self, aspect=1.0, pan=(0.0, 0.0), zoom=(1.0, 1.0),
                 zmin=1e-5, zmax=1e5,
                 xmin=None, xmax=None,
                 ymin=None, ymax=None,
                 ):
        """
        Initialize the transform.

        Parameters
        ----------

        aspect : float (default is None)
           Indicate what is the aspect ratio of the object displayed. This is
           necessary to convert pixel drag move in object space coordinates.

        pan : float, float (default is 0, 0)
           Initial translation

        zoom : float, float (default is 1)
           Initial zoom level

        zmin : float (default is 0.01)
           Minimum zoom level

        zmax : float (default is 1000)
           Maximum zoom level
        """

        self._aspect = aspect
        self._zmin = zmin
        self._zmax = zmax
        self._xmin = xmin
        self._xmax = xmax
        self._ymin = ymin
        self._ymax = ymax

        self._zoom_to_pointer = True

        # Canvas this transform is attached to
        self._canvas = None
        self._canvas_aspect = np.ones(2)
        self._width = 1
        self._height = 1

        self._create_pan_and_zoom(_as_array(pan), _as_array(zoom))

        # Programs using this transform
        self._programs = []

    def _create_pan_and_zoom(self, pan, zoom):
        self._pan = np.array(pan)
        self._zoom = np.array(zoom)
        self._zoom_coeff = self._default_zoom_coeff
        self._wheel_coeff = self._default_wheel_coeff

    # Various properties
    # -------------------------------------------------------------------------

    @property
    def zoom_to_pointer(self):
        return self._zoom_to_pointer

    @zoom_to_pointer.setter
    def zoom_to_pointer(self, value):
        self._zoom_to_pointer = value

    @property
    def is_attached(self):
        """Whether transform is attached to a canvas."""
        return self._canvas is not None

    @property
    def aspect(self):
        """Aspect (width/height)."""
        return self._aspect

    @aspect.setter
    def aspect(self, value):
        """Aspect (width/height)."""
        self._aspect = value

    # xmin/xmax
    # -------------------------------------------------------------------------

    @property
    def xmin(self):
        return self._xmin

    @xmin.setter
    def xmin(self, value):
        if self._xmax is not None:
            self._xmin = min(value, self._xmax)
        else:
            self._xmin = value

    @property
    def xmax(self):
        return self._xmax

    @xmax.setter
    def xmax(self, value):
        if self._xmin is not None:
            self._xmax = max(value, self._xmin)
        else:
            self._xmax = value

    # ymin/ymax
    # -------------------------------------------------------------------------

    @property
    def ymin(self):
        return self._ymin

    @ymin.setter
    def ymin(self, value):
        if self._ymax is not None:
            self._ymin = min(value, self._ymax)
        else:
            self._ymin = value

    @property
    def ymax(self):
        return self._ymax

    @ymax.setter
    def ymax(self, value):
        if self._ymin is not None:
            self._ymax = max(value, self._ymin)
        else:
            self._ymax = value

    # zmin/zmax
    # -------------------------------------------------------------------------

    @property
    def zmin(self):
        """Minimum zoom level."""
        return self._zmin

    @zmin.setter
    def zmin(self, value):
        """Minimum zoom level."""
        self._zmin = min(value, self._zmax)

    @property
    def zmax(self):
        """Maximal zoom level."""
        return self._zmax

    @zmax.setter
    def zmax(self, value):
        """Maximal zoom level."""
        self._zmax = max(value, self._zmin)

    # Internal methods
    # -------------------------------------------------------------------------

    def _apply_pan_zoom(self):
        zoom = self._zoom_aspect()
        for program in self._programs:
            program["u_pan"] = self._pan
            program["u_zoom"] = zoom

    def _zoom_aspect(self, zoom=None):
        if zoom is None:
            zoom = self._zoom
        zoom = _as_array(zoom)
        if self._aspect is not None:
            aspect = self._canvas_aspect * self._aspect
        else:
            aspect = 1.
        return zoom * aspect

    def _normalize(self, x_y, restrict_to_box=True):
        x_y = np.asarray(x_y, dtype=np.float32)
        size = np.array([self._width, self._height], dtype=np.float32)
        pos = x_y / (size / 2.) - 1
        return pos

    def _constrain_pan(self):
        """Constrain bounding box."""
        if self._xmin is not None and self._xmax is not None:
            p0 = self._xmin + 1. / self._zoom[0]
            p1 = self._xmax - 1. / self._zoom[0]
            p0, p1 = min(p0, p1), max(p0, p1)
            self._pan[0] = np.clip(self._pan[0], p0, p1)

        if self._ymin is not None and self._ymax is not None:
            p0 = self._ymin + 1. / self._zoom[1]
            p1 = self._ymax - 1. / self._zoom[1]
            p0, p1 = min(p0, p1), max(p0, p1)
            self._pan[1] = np.clip(self._pan[1], p0, p1)

    def _constrain_zoom(self):
        """Constrain bounding box."""
        if self._xmin is not None:
            self._zoom[0] = max(self._zoom[0],
                                1. / (self._pan[0] - self._xmin))
        if self._xmax is not None:
            self._zoom[0] = max(self._zoom[0],
                                1. / (self._xmax - self._pan[0]))

        if self._ymin is not None:
            self._zoom[1] = max(self._zoom[1],
                                1. / (self._pan[1] - self._ymin))
        if self._ymax is not None:
            self._zoom[1] = max(self._zoom[1],
                                1. / (self._ymax - self._pan[1]))

    # Pan and zoom
    # -------------------------------------------------------------------------

    @property
    def pan(self):
        """Pan translation."""
        return self._pan

    @pan.setter
    def pan(self, value):
        """Pan translation."""
        assert len(value) == 2
        self._pan[:] = value
        self._constrain_pan()
        self._apply_pan_zoom()

    @property
    def zoom(self):
        """Zoom level."""
        return self._zoom

    @zoom.setter
    def zoom(self, value):
        """Zoom level."""
        if isinstance(value, (int, float)):
            value = (value, value)
        assert len(value) == 2
        self._zoom = np.clip(value, self._zmin, self._zmax)
        if not self.is_attached:
            return

        # Constrain bounding box.
        self._constrain_pan()
        self._constrain_zoom()

        self._apply_pan_zoom()

    def _do_pan(self, d):

        dx, dy = d

        pan_x, pan_y = self.pan
        zoom_x, zoom_y = self._zoom_aspect(self._zoom)

        self.pan = (pan_x + dx / zoom_x,
                    pan_y + dy / zoom_y)

        self._canvas.update()

    def _do_zoom(self, d, p, c=1.):
        dx, dy = d
        x0, y0 = p

        pan_x, pan_y = self._pan
        zoom_x, zoom_y = self._zoom
        zoom_x_new, zoom_y_new = (zoom_x * math.exp(c * self._zoom_coeff * dx),
                                  zoom_y * math.exp(c * self._zoom_coeff * dy))

        zoom_x_new = max(min(zoom_x_new, self._zmax), self._zmin)
        zoom_y_new = max(min(zoom_y_new, self._zmax), self._zmin)

        self.zoom = zoom_x_new, zoom_y_new

        if self._zoom_to_pointer:
            zoom_x, zoom_y = self._zoom_aspect((zoom_x,
                                                zoom_y))
            zoom_x_new, zoom_y_new = self._zoom_aspect((zoom_x_new,
                                                        zoom_y_new))

            self.pan = (pan_x - x0 * (1. / zoom_x - 1. / zoom_x_new),
                        pan_y + y0 * (1. / zoom_y - 1. / zoom_y_new))

        self._canvas.update()

    # Event callbacks
    # -------------------------------------------------------------------------

    def on_resize(self, event):
        """Resize event."""

        self._width = float(event.size[0])
        self._height = float(event.size[1])
        aspect = self._width / self._height
        if aspect > 1.0:
            self._canvas_aspect = np.array([1.0 / aspect, 1.0])
        else:
            self._canvas_aspect = np.array([1.0, aspect / 1.0])

        # Update zoom level
        self.zoom = self._zoom

    def on_mouse_move(self, event):
        """Pan and zoom with the mouse."""

        if event.is_dragging and not event.modifiers:
            x0, y0 = self._normalize(event.press_event.pos)
            x1, y1 = self._normalize(event.last_event.pos, False)
            x, y = self._normalize(event.pos, False)
            dx, dy = x - x1, -(y - y1)
            if event.button == 1:
                self._do_pan((dx, dy))
            elif event.button == 2:
                c = np.sqrt(self._width) * .03
                self._do_zoom((dx, dy), (x0, y0), c=c)

    def on_mouse_wheel(self, event):
        """Zoom."""
        dx = np.sign(event.delta[1]) * self._wheel_coeff
        # Zoom toward the mouse pointer.
        x0, y0 = self._normalize(event.pos)
        self._do_zoom((dx, dx), (x0, y0))

    _arrows = ('Left', 'Right', 'Up', 'Down')
    _pm = ('+', '-')

    def on_key_press(self, event):
        # Zooming with the keyboard.
        key = event.key
        if event.modifiers:
            return

        # Pan.
        if key in self._arrows:
            k = .1 / self.zoom
            if key == 'Left':
                self.pan += (+k[0], +0)
            elif key == 'Right':
                self.pan += (-k[0], +0)
            elif key == 'Down':
                self.pan += (+0, +k[1])
            elif key == 'Up':
                self.pan += (+0, -k[1])
            self._canvas.update()

        # Zoom.
        if key in self._pm:
            k = .05
            if key == '-':
                self.zoom *= (1. - k)
            elif key == '+':
                self.zoom *= (1. + k)
            self._canvas.update()

        # Reset with 'R'.
        if key == 'R':
            self.pan = (0., 0.)
            self.zoom = 1.
            self._canvas.update()

    # Canvas methods
    # -------------------------------------------------------------------------

    def add(self, programs):
        """ Attach programs to this tranform """

        if not isinstance(programs, (list, tuple)):
            programs = [programs]

        for program in programs:
            self._programs.append(program)

        self._apply_pan_zoom()

    def attach(self, canvas):
        """ Attach this tranform to a canvas """

        self._canvas = canvas
        self._width = float(canvas.size[0])
        self._height = float(canvas.size[1])

        aspect = self._width / self._height
        if aspect > 1.0:
            self._canvas_aspect = np.array([1.0 / aspect, 1.0])
        else:
            self._canvas_aspect = np.array([1.0, aspect / 1.0])

        canvas.connect(self.on_resize)
        canvas.connect(self.on_mouse_wheel)
        canvas.connect(self.on_mouse_move)
        canvas.connect(self.on_key_press)


class PanZoomGrid(PanZoom):
    """Pan & zoom transform for a grid view.

    This is used in a grid view with independent per-subplot pan & zoom.

    The currently-active subplot depends on where the cursor was when
    the mouse was clicked.

    Interactivity
    -------------

    Pan:

    * Mouse : left-click and move (drag movement)
    * Keyboard : arrows

    Subplot zoom:

    * Mouse : wheel or right-click and move (drag movement)
    * Keyboard : + and -

    Global zoom:

    * Mouse : Alt + wheel

    Subplot reset:

    * Keyboard : R

    Global reset:

    * Keyboard : Alt + R

    """
    _index = (0, 0)  # current index of the box being pan/zoom-ed

    def __init__(self, *args, **kwargs):
        n_rows = kwargs.pop('n_rows')
        self._n_rows = n_rows
        super(PanZoomGrid, self).__init__(*args, **kwargs)

    # Grid properties
    # -------------------------------------------------------------------------

    @property
    def n_rows(self):
        assert self._n_rows is not None
        return self._n_rows

    @n_rows.setter
    def n_rows(self, value):
        self._n_rows = int(value)
        assert self._n_rows >= 1
        if self._n_rows > 16:
            raise RuntimeError("There cannot be more than 16x16 subplots. "
                               "The limitation comes from the maximum "
                               "uniform array size used by panzoom. "
                               "But you can try to increase the number 256 in "
                               "'plot/glsl/grid.glsl'.")
        self._create_pan_and_zoom((0., 0.), (1., 1.))

    # Pan and zoom
    # -------------------------------------------------------------------------

    def _create_pan_and_zoom(self, pan, zoom):
        pan = _as_array(pan)
        zoom = _as_array(zoom)
        self._pan_matrix = np.empty((self._n_rows, self._n_rows, 2))
        self._pan_matrix[..., :] = pan[None, None, :]

        self._zoom_matrix = np.empty((self._n_rows, self._n_rows, 2))
        self._zoom_matrix[..., :] = zoom[None, None, :]

        # The zoom coefficient for mouse zoom should be proportional
        # to the subplot size.
        c = 3. / np.sqrt(self._n_rows)
        self._zoom_coeff = self._default_zoom_coeff * c
        self._wheel_coeff = self._default_wheel_coeff

    def _set_current_box(self, pos):
        self._index = self._get_box(pos)

    @property
    def _pan(self):
        i, j = self._index
        return self._pan_matrix[i, j, :]

    @_pan.setter
    def _pan(self, value):
        i, j = self._index
        self._pan_matrix[i, j, :] = value

    @property
    def _zoom(self):
        i, j = self._index
        return self._zoom_matrix[i, j, :]

    @_zoom.setter
    def _zoom(self, value):
        i, j = self._index
        self._zoom_matrix[i, j, :] = value

    @property
    def zoom_matrix(self):
        return self._zoom_matrix

    def _apply_pan_zoom(self):
        pan = self._pan
        zoom = self._zoom_aspect(self._zoom)
        i, j = self._index
        box = int(i * self._n_rows + j)
        value = (pan[0], pan[1], zoom[0], zoom[1])
        for program in self._programs:
            program["u_pan_zoom[{0:d}]".format(box)] = value

    # Internal methods
    # -------------------------------------------------------------------------

    def _constrain_pan(self):
        """Constrain bounding box."""
        if self._xmin is not None and self._xmax is not None:
            p0 = (self._xmin + 1. / self._zoom[0]) / self._n_rows
            p1 = (self._xmax - 1. / self._zoom[0]) / self._n_rows
            p0, p1 = min(p0, p1), max(p0, p1)
            self._pan[0] = np.clip(self._pan[0], p0, p1)

        if self._ymin is not None and self._ymax is not None:
            p0 = (self._ymin + 1. / self._zoom[1]) / self._n_rows
            p1 = (self._ymax - 1. / self._zoom[1]) / self._n_rows
            p0, p1 = min(p0, p1), max(p0, p1)
            self._pan[1] = np.clip(self._pan[1], p0, p1)

    def _get_box(self, x_y):
        x0, y0 = x_y

        x0 /= self._width
        y0 /= self._height

        x0 *= self._n_rows
        y0 *= self._n_rows

        return (math.floor(y0), math.floor(x0))

    def _normalize(self, x_y, restrict_to_box=True):
        x0, y0 = x_y

        x0 /= self._width
        y0 /= self._height

        x0 *= self._n_rows
        y0 *= self._n_rows

        if restrict_to_box:
            x0 = x0 % 1
            y0 = y0 % 1

        x0 = -(1 - 2 * x0)
        y0 = -(1 - 2 * y0)

        x0 /= self._n_rows
        y0 /= self._n_rows

        return x0, y0

    def _initialize_pan_zoom(self):
        # Initialize and set the uniform array.
        # NOTE: 256 is the maximum size used for this uniform array.
        # This corresponds to a hard-limit of 16x16 subplots.
        self._u_pan_zoom = np.zeros((256, 4),
                                    dtype=np.float32)
        self._u_pan_zoom[:, 2:] = 1.
        for program in self._programs:
            program["u_pan_zoom"] = self._u_pan_zoom

    def _global_pan_zoom(self, pan, zoom):
        # Reinitialize the pan zoom matrix.
        self._create_pan_and_zoom(pan, zoom)
        value = tuple(pan) + tuple(zoom)
        for program in self._programs:
            # Update all boxes one by one.
            # TODO OPTIM: update the uniform array at once.
            # But it doesn't seem to work (VisPy bug?).
            for box in range(self._n_rows * self._n_rows):
                program["u_pan_zoom[{0:d}]".format(box)] = value

    def _reset(self):
        self._global_pan_zoom((0., 0.), (1., 1.))

    # Event callbacks
    # -------------------------------------------------------------------------

    def on_mouse_move(self, event):
        # Set box index as a function of the press position.
        if event.is_dragging:
            self._set_current_box(event.press_event.pos)
        super(PanZoomGrid, self).on_mouse_move(event)

    def on_mouse_wheel(self, event):
        # Set box index as a function of the press position.
        self._set_current_box(event.pos)
        super(PanZoomGrid, self).on_mouse_wheel(event)

        modifiers = event.modifiers
        alt = 'Alt' in modifiers

        # Global zoom.
        if alt:
            self._global_pan_zoom((0., 0.), self._zoom)
            self._canvas.update()

    def on_key_press(self, event):
        super(PanZoomGrid, self).on_key_press(event)

        key = event.key
        modifiers = event.modifiers
        alt = 'Alt' in modifiers

        # Reset with 'R'.
        if key == 'R' and alt:
            self._reset()
            self._canvas.update()

    # Canvas methods
    # -------------------------------------------------------------------------

    def add(self, programs):
        """ Attach programs to this tranform """
        if not isinstance(programs, (list, tuple)):
            programs = [programs]
        for program in programs:
            self._programs.append(program)
        self._initialize_pan_zoom()
        self._apply_pan_zoom()
