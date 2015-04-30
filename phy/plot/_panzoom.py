# -*- coding: utf-8 -*-

"""Pan & zoom transform."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import math

import numpy as np

from vispy import gloo

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

    * Mouse : click and move (drag movement)
    * Keyboard : arrows

    Zoom:

    * Mouse : wheel
    * Keyboard : + and -

    Reset:

    * Keyboard : R

    """

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
        self._create_pan_and_zoom(pan, zoom)
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

        # Programs using this transform
        self._programs = []

    def _create_pan_and_zoom(self, pan, zoom):
        self._pan = np.array(pan)
        self._zoom = np.array(zoom)

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

    def _apply_pan(self):
        for program in self._programs:
            program["u_pan"] = self._pan

    def _zoom_aspect(self, zoom=None):
        if zoom is None:
            zoom = self._zoom
        zoom = _as_array(zoom)
        if self._aspect is not None:
            aspect = self._canvas_aspect * self._aspect
        else:
            aspect = np.ones(2)
        return zoom * aspect

    def _apply_zoom(self):
        zoom = self._zoom_aspect()
        for program in self._programs:
            program["u_zoom"] = zoom

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
        self._apply_pan()

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

        self._apply_pan()
        self._apply_zoom()

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
        """Drag."""
        if event.is_dragging and not event.modifiers:
            if event.button == 1:
                x0, y0 = self._normalize(event.press_event.pos)
                x1, y1 = self._normalize(event.last_event.pos, False)
                x, y = self._normalize(event.pos, False)
                dx, dy = x - x1, -(y - y1)

                pan_x, pan_y = self.pan
                zoom_x, zoom_y = self._zoom_aspect(self._zoom)

                self.pan = (pan_x + dx / zoom_x,
                            pan_y + dy / zoom_y)

                self._canvas.update()
            elif event.button == 2:
                x0, y0 = self._normalize(event.press_event.pos)
                x1, y1 = self._normalize(event.last_event.pos, False)
                x, y = self._normalize(event.pos, False)
                dx, dy = x - x1, -(y - y1)
                z_old = self.zoom
                self.zoom = z_old * np.exp(2. * np.array([dx, dy]))
                z_new = self.zoom
                self.pan += -np.array([x0, -y0]) * (1. / z_old - 1. / z_new)
                self._canvas.update()

    def on_mouse_wheel(self, event):
        """Zoom."""
        dx = np.sign(event.delta[1]) * .05
        # Zoom toward the mouse pointer.
        x0, y0 = self._normalize(event.pos)
        pan_x, pan_y = self._pan
        zoom_x, zoom_y = self._zoom
        zoom_x_new, zoom_y_new = (zoom_x * math.exp(2.5 * dx),
                                  zoom_y * math.exp(2.5 * dx))

        zoom_x_new = max(min(zoom_x_new, self._zmax), self._zmin)
        zoom_y_new = max(min(zoom_y_new, self._zmax), self._zmin)

        self.zoom = zoom_x_new, zoom_y_new

        if self._zoom_to_pointer:
            # if self._aspect is not None:
            #     aspect = self._canvas_aspect * self._aspect
            # else:
            #     aspect = np.ones(2)
            # zoom_x *= aspect[0]
            # zoom_y *= aspect[1]
            # zoom_x_new *= aspect[0]
            # zoom_y_new *= aspect[1]
            zoom_x, zoom_y = self._zoom_aspect((zoom_x,
                                                zoom_y))
            zoom_x_new, zoom_y_new = self._zoom_aspect((zoom_x_new,
                                                        zoom_y_new))

            self.pan = (pan_x - x0 * (1. / zoom_x - 1. / zoom_x_new),
                        pan_y + y0 * (1. / zoom_y - 1. / zoom_y_new))

        self._canvas.update()

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
                self.pan += (+k, +0)
            elif key == 'Right':
                self.pan += (-k, +0)
            elif key == 'Down':
                self.pan += (+0, +k)
            elif key == 'Up':
                self.pan += (+0, -k)
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

    def add(self, programs):
        """ Attach programs to this tranform """

        if not isinstance(programs, (list, tuple)):
            programs = [programs]

        for program in programs:
            self._programs.append(program)

        self._apply_pan()
        self._apply_zoom()

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
    _n_rows = 1
    _index = (0, 0)  # current index of the box being pan/zoom-ed

    # def __init__(self, *args, **kwargs):
    #     super(PanZoomGrid, self).__init__(*args, **kwargs)

    def _create_u_pan_and_zoom(self):
        u_pan = np.array(self._pan)
        u_zoom = np.array([self._zoom, self._zoom])

        self._pan_zoom_matrix = np.empty((self._n_rows, self._n_rows, 4),
                                         dtype=np.float32)
        self._pan_zoom_matrix[..., :2] = u_pan[None, None, :]
        self._pan_zoom_matrix[..., 2:] = u_zoom[None, None, :]

    @property
    def _u_pan(self):
        i, j = self._index
        return self._pan_zoom_matrix[i, j, :2]

    @_u_pan.setter
    def _u_pan(self, value):
        i, j = self._index
        self._pan_zoom_matrix[i, j, :2] = value

    @property
    def _u_zoom(self):
        i, j = self._index
        return self._pan_zoom_matrix[i, j, 2:]

    @_u_zoom.setter
    def _u_zoom(self, value):
        i, j = self._index
        self._pan_zoom_matrix[i, j, 2:] = value

    def add(self, programs):
        """ Attach programs to this tranform """

        if not isinstance(programs, (list, tuple)):
            programs = [programs]

        for program in programs:
            self._programs.append(program)

        # Initialize and set the texture.
        self._create_pan_zoom_texture(1)
        # for program in self._programs:
        #     program["u_pan_zoom"] = gloo.Texture2D(self._tex)

        self._apply_pan()
        self._apply_zoom()

    def _normalize_pan(self, pan):
        return (_as_array(pan) + 1.) / 10. * 255

    def _normalize_zoom(self, zoom):
        return _as_array(zoom) / 10. * 255

    def _apply_pan(self):
        i, j = self._index
        self._tex[i, j, :2] = self._normalize_pan(self._pan)
        self._tex[i, j, 2:] = self._normalize_zoom(self._zoom)
        for program in self._programs:
            program["u_pan_zoom"].set_data(self._tex)

    def _apply_zoom(self):
        self._apply_pan()

    def _create_pan_zoom_texture(self, n_rows):
        self._create_u_pan_and_zoom()

        shape = (n_rows, n_rows, 4)
        self._tex = np.zeros(shape, dtype=np.uint8)
        self._tex[..., :2] = self._normalize_pan(self._pan)
        self._tex[..., 2:] = self._normalize_zoom(self._zoom)
        for program in self._programs:
            program["u_pan_zoom"] = gloo.Texture2D(self._tex)

    def _set_current_box(self, pos):
        self._index = self._get_box(pos)

    # def on_mouse_press(self, event):
    #     self._set_current_box(event.pos)

    def on_mouse_move(self, event):
        if event.is_dragging:
            # Set box index as a function of the press position.
            self._set_current_box(event.press_event.pos)
        super(PanZoomGrid, self).on_mouse_move(event)

    def on_mouse_wheel(self, event):
        # Set box index as a function of the press position.
        self._set_current_box(event.pos)
        super(PanZoomGrid, self).on_mouse_wheel(event)

    @property
    def n_rows(self):
        assert self._n_rows is not None
        return self._n_rows

    @n_rows.setter
    def n_rows(self, value):
        self._n_rows = int(value)
        assert self._n_rows >= 1
        self._create_pan_zoom_texture(value)

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
