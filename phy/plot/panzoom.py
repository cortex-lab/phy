# -*- coding: utf-8 -*-

"""Pan & zoom transform."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import math

import numpy as np

from .base import BaseInteract
from .transform import Translate, Scale, pixels_to_ndc
from phy.utils._types import _as_array


#------------------------------------------------------------------------------
# PanZoom class
#------------------------------------------------------------------------------

class PanZoom(BaseInteract):
    """Pan and zoom interact.

    To use it:

    ```python
    # Create a visual.
    visual = MyVisual(...)
    visual.set_data(...)

    # Attach the visual to the canvas.
    canvas = BaseCanvas()
    visual.attach(canvas, 'PanZoom')

    # Create and attach the PanZoom interact.
    pz = PanZoom()
    pz.attach(canvas)

    canvas.show()
    ```

    """

    _default_zoom_coeff = 1.5
    _default_wheel_coeff = .1
    _arrows = ('Left', 'Right', 'Up', 'Down')
    _pm = ('+', '-')

    def __init__(self,
                 aspect=1.0,
                 pan=(0.0, 0.0), zoom=(1.0, 1.0),
                 zmin=1e-5, zmax=1e5,
                 xmin=None, xmax=None,
                 ymin=None, ymax=None,
                 constrain_bounds=None,
                 pan_var_name='u_pan',
                 zoom_var_name='u_zoom',
                 ):
        super(PanZoom, self).__init__()

        if constrain_bounds:
            assert xmin is None
            assert ymin is None
            assert xmax is None
            assert ymax is None
            xmin, ymin, xmax, ymax = constrain_bounds

        self.pan_var_name = pan_var_name
        self.zoom_var_name = zoom_var_name

        self._aspect = aspect

        self._zmin = zmin
        self._zmax = zmax
        self._xmin = xmin
        self._xmax = xmax
        self._ymin = ymin
        self._ymax = ymax

        self._pan = np.array(pan)
        self._zoom = np.array(zoom)

        self._zoom_coeff = self._default_zoom_coeff
        self._wheel_coeff = self._default_wheel_coeff

        self._zoom_to_pointer = True
        self._canvas_aspect = np.ones(2)

    def get_shader_declarations(self):
        return ('uniform vec2 {};\n'.format(self.pan_var_name) +
                'uniform vec2 {};\n'.format(self.zoom_var_name)), ''

    def get_transforms(self):
        return [Translate(translate=self.pan_var_name),
                Scale(scale=self.zoom_var_name)]

    def update_program(self, program):
        zoom = self._zoom_aspect()
        program[self.pan_var_name] = self._pan
        program[self.zoom_var_name] = zoom

    # Various properties
    # -------------------------------------------------------------------------

    @property
    def aspect(self):
        """Aspect (width/height)."""
        return self._aspect

    @aspect.setter
    def aspect(self, value):
        self._aspect = value

    # xmin/xmax
    # -------------------------------------------------------------------------

    @property
    def xmin(self):
        """Minimum x allowed for pan."""
        return self._xmin

    @xmin.setter
    def xmin(self, value):
        self._xmin = (np.minimum(value, self._xmax)
                      if self._xmax is not None else value)

    @property
    def xmax(self):
        """Maximum x allowed for pan."""
        return self._xmax

    @xmax.setter
    def xmax(self, value):
        self._xmax = (np.maximum(value, self._xmin)
                      if self._xmin is not None else value)

    # ymin/ymax
    # -------------------------------------------------------------------------

    @property
    def ymin(self):
        """Minimum y allowed for pan."""
        return self._ymin

    @ymin.setter
    def ymin(self, value):
        self._ymin = (min(value, self._ymax)
                      if self._ymax is not None else value)

    @property
    def ymax(self):
        """Maximum y allowed for pan."""
        return self._ymax

    @ymax.setter
    def ymax(self, value):
        self._ymax = (max(value, self._ymin)
                      if self._ymin is not None else value)

    # zmin/zmax
    # -------------------------------------------------------------------------

    @property
    def zmin(self):
        """Minimum zoom level."""
        return self._zmin

    @zmin.setter
    def zmin(self, value):
        self._zmin = min(value, self._zmax)

    @property
    def zmax(self):
        """Maximal zoom level."""
        return self._zmax

    @zmax.setter
    def zmax(self, value):
        self._zmax = max(value, self._zmin)

    # Internal methods
    # -------------------------------------------------------------------------

    def _zoom_aspect(self, zoom=None):
        zoom = zoom if zoom is not None else self._zoom
        zoom = _as_array(zoom)
        aspect = (self._canvas_aspect * self._aspect
                  if self._aspect is not None else 1.)
        return zoom * aspect

    def _normalize(self, pos):
        return pixels_to_ndc(pos, size=self.size)

    def _constrain_pan(self):
        """Constrain bounding box."""
        if self.xmin is not None and self.xmax is not None:
            p0 = self.xmin + 1. / self._zoom[0]
            p1 = self.xmax - 1. / self._zoom[0]
            p0, p1 = min(p0, p1), max(p0, p1)
            self._pan[0] = np.clip(self._pan[0], p0, p1)

        if self.ymin is not None and self.ymax is not None:
            p0 = self.ymin + 1. / self._zoom[1]
            p1 = self.ymax - 1. / self._zoom[1]
            p0, p1 = min(p0, p1), max(p0, p1)
            self._pan[1] = np.clip(self._pan[1], p0, p1)

    def _constrain_zoom(self):
        """Constrain bounding box."""
        if self.xmin is not None:
            self._zoom[0] = max(self._zoom[0],
                                1. / (self._pan[0] - self.xmin))
        if self.xmax is not None:
            self._zoom[0] = max(self._zoom[0],
                                1. / (self.xmax - self._pan[0]))

        if self.ymin is not None:
            self._zoom[1] = max(self._zoom[1],
                                1. / (self._pan[1] - self.ymin))
        if self.ymax is not None:
            self._zoom[1] = max(self._zoom[1],
                                1. / (self.ymax - self._pan[1]))

    # Pan and zoom
    # -------------------------------------------------------------------------

    @property
    def pan(self):
        """Pan translation."""
        return list(self._pan)

    @pan.setter
    def pan(self, value):
        """Pan translation."""
        assert len(value) == 2
        self._pan[:] = value
        self._constrain_pan()
        self.update()

    @property
    def zoom(self):
        """Zoom level."""
        return list(self._zoom)

    @zoom.setter
    def zoom(self, value):
        """Zoom level."""
        if isinstance(value, (int, float)):
            value = (value, value)
        assert len(value) == 2
        self._zoom = np.clip(value, self._zmin, self._zmax)

        # Constrain bounding box.
        self._constrain_pan()
        self._constrain_zoom()

        self.update()

    def pan_delta(self, d):
        """Pan the view by a given amount."""
        dx, dy = d

        pan_x, pan_y = self.pan
        zoom_x, zoom_y = self._zoom_aspect(self._zoom)

        self.pan = (pan_x + dx / zoom_x, pan_y + dy / zoom_y)
        self.update()

    def zoom_delta(self, d, p=(0., 0.), c=1.):
        """Zoom the view by a given amount."""
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
                        pan_y - y0 * (1. / zoom_y - 1. / zoom_y_new))

        self.update()

    # Event callbacks
    # -------------------------------------------------------------------------

    keyboard_shortcuts = {
        'pan': ('left click and drag', 'arrows'),
        'zoom': ('right click and drag', '+', '-'),
        'reset': 'r',
    }

    def _set_canvas_aspect(self):
        w, h = self.size
        aspect = w / max(float(h), 1.)
        if aspect > 1.0:
            self._canvas_aspect = np.array([1.0 / aspect, 1.0])
        else:
            self._canvas_aspect = np.array([1.0, aspect / 1.0])

    def _zoom_keyboard(self, key):
        k = .05
        if key == '-':
            k = -k
        self.zoom_delta((k, k), (0, 0))

    def _pan_keyboard(self, key):
        k = .1 / np.asarray(self.zoom)
        if key == 'Left':
            self.pan_delta((+k[0], +0))
        elif key == 'Right':
            self.pan_delta((-k[0], +0))
        elif key == 'Down':
            self.pan_delta((+0, +k[1]))
        elif key == 'Up':
            self.pan_delta((+0, -k[1]))
        self.update()

    def reset(self):
        """Reset the view."""
        self.pan = (0., 0.)
        self.zoom = 1.
        self.update()

    def on_resize(self, event):
        """Resize event."""
        super(PanZoom, self).on_resize(event)
        self._set_canvas_aspect()
        # Update zoom level
        self.zoom = self._zoom

    def on_mouse_move(self, event):
        """Pan and zoom with the mouse."""
        super(PanZoom, self).on_mouse_move(event)
        if event.modifiers:
            return
        if event.is_dragging:
            x0, y0 = self._normalize(event.press_event.pos)
            x1, y1 = self._normalize(event.last_event.pos)
            x, y = self._normalize(event.pos)
            dx, dy = x - x1, y - y1
            if event.button == 1:
                self.pan_delta((dx, dy))
            elif event.button == 2:
                c = np.sqrt(self.size[0]) * .03
                self.zoom_delta((dx, dy), (x0, y0), c=c)

    def on_mouse_wheel(self, event):
        """Zoom with the mouse wheel."""
        super(PanZoom, self).on_mouse_wheel(event)
        if event.modifiers:
            return
        dx = np.sign(event.delta[1]) * self._wheel_coeff
        # Zoom toward the mouse pointer.
        x0, y0 = self._normalize(event.pos)
        self.zoom_delta((dx, dx), (x0, y0))

    def on_key_press(self, event):
        """Pan and zoom with the keyboard."""
        super(PanZoom, self).on_key_press(event)

        # Zooming with the keyboard.
        key = event.key
        if event.modifiers:
            return

        # Pan.
        if key in self._arrows:
            self._pan_keyboard(key)

        # Zoom.
        if key in self._pm:
            self._zoom_keyboard(key)

        # Reset with 'R'.
        if key == 'R':
            self.reset()

    # Canvas methods
    # -------------------------------------------------------------------------

    def attach(self, canvas):
        """Attach this interact to a canvas."""
        super(PanZoom, self).attach(canvas)
        self._set_canvas_aspect()