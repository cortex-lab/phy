# -*- coding: utf-8 -*-

"""Pan & zoom transform."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import math

import numpy as np


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

    def __init__(self, aspect=1.0, pan=(0.0, 0.0), zoom=1.0,
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
        self._pan = np.array(pan)
        self._zoom = zoom
        self._zmin = zmin
        self._zmax = zmax
        self._xmin = xmin
        self._xmax = xmax
        self._ymin = ymin
        self._ymax = ymax

        self._zoom_to_pointer = True
        self._n_rows = 1

        # Canvas this transform is attached to
        self._canvas = None
        self._canvas_aspect = np.ones(2)
        self._width = 1
        self._height = 1

        # Programs using this transform
        self._u_pan = pan
        self._u_zoom = np.array([zoom, zoom])
        self._programs = []

    @property
    def zoom_to_pointer(self):
        return self._zoom_to_pointer

    @zoom_to_pointer.setter
    def zoom_to_pointer(self, value):
        self._zoom_to_pointer = value

    @property
    def n_rows(self):
        return self._n_rows

    @n_rows.setter
    def n_rows(self, value):
        self._n_rows = value

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

    @property
    def pan(self):
        """Pan translation."""
        return self._pan

    @pan.setter
    def pan(self, value):
        """Pan translation."""
        self._pan = np.asarray(value)

        # Constrain bounding box.
        s = 1. / self._zoom
        if self._xmin is not None:
            self._pan[0] = max(self._pan[0], self._xmin + s)
        if self._xmax is not None:
            self._pan[0] = min(self._pan[0], self._xmax - s)

        if self._ymin is not None:
            self._pan[1] = max(self._pan[1], self._ymin + s)
        if self._ymax is not None:
            self._pan[1] = min(self._pan[1], self._ymax - s)

        self._u_pan = self._pan
        for program in self._programs:
            program["u_pan"] = self._u_pan

    @property
    def zoom(self):
        """Zoom level."""
        return self._zoom

    @zoom.setter
    def zoom(self, value):
        """Zoom level."""

        self._zoom = max(min(value, self._zmax), self._zmin)
        if not self.is_attached:
            return

        aspect = np.array([1.0, 1.0])
        if self._aspect is not None:
            aspect = self._canvas_aspect * self._aspect

        self._u_zoom = self._zoom * aspect
        for program in self._programs:
            program["u_zoom"] = self._u_zoom

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

    def _normalize(self, x_y):
        x_y = np.asarray(x_y, dtype=np.float32)
        size = np.array([self._width, self._height], dtype=np.float32)
        pos = x_y / (size / 2.) - 1
        return pos

    def _normalize_grid(self, x_y):
        x0, y0 = x_y

        x0 /= self._width
        y0 /= self._height

        x0 *= self._n_rows
        y0 *= self._n_rows

        x0 = x0 % 1
        y0 = y0 % 1

        x0 = -(1 - 2 * x0)
        y0 = -(1 - 2 * y0)

        x0 /= self._n_rows
        y0 /= self._n_rows

        return x0, y0

    def on_mouse_move(self, event):
        """Drag."""

        if event.is_dragging and not event.modifiers:
            x0, y0 = self._normalize(event.press_event.pos)
            x1, y1 = self._normalize(event.last_event.pos)
            x, y = self._normalize(event.pos)
            dx, dy = x - x1, -(y - y1)

            pan_x, pan_y = self.pan
            zoom_x, zoom_y = self._u_zoom

            self.pan = (pan_x + dx / zoom_x,
                        pan_y + dy / zoom_y)

            self._canvas.update()

    def on_mouse_wheel(self, event):
        """Zoom."""

        dx = np.sign(event.delta[1]) * .05
        # Zoom toward the mouse pointer in the grid view.
        x0, y0 = self._normalize_grid(event.pos)

        pan_x, pan_y = self.pan
        zoom_x = zoom_y = self.zoom
        zoom_x_new, zoom_y_new = (zoom_x * math.exp(2.5 * dx),
                                  zoom_y * math.exp(2.5 * dx))

        zoom_x_new = max(min(zoom_x_new, self._zmax), self._zmin)
        zoom_y_new = max(min(zoom_y_new, self._zmax), self._zmin)

        self.zoom = zoom_x_new

        if self._zoom_to_pointer:
            aspect = np.array([1.0, 1.0])
            if self._aspect is not None:
                aspect = self._canvas_aspect * self._aspect
            zoom_x *= aspect[0]
            zoom_y *= aspect[1]
            zoom_x_new *= aspect[0]
            zoom_y_new *= aspect[1]

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
            program["u_zoom"] = self._u_zoom
            program["u_pan"] = self._u_pan

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

        aspect = np.array([1.0, 1.0])
        if self._aspect is not None:
            aspect = self._canvas_aspect * self._aspect
        self._u_zoom = self._zoom * aspect

        canvas.connect(self.on_resize)
        canvas.connect(self.on_mouse_wheel)
        canvas.connect(self.on_mouse_move)
        canvas.connect(self.on_key_press)
