# -*- coding: utf-8 -*-

"""Base VisPy classes."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging

from vispy import gloo
from vispy.app import Canvas

from .utils import _create_program

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Base spike visual
#------------------------------------------------------------------------------

class BaseCanvas(Canvas):
    def __init__(self, *args, **kwargs):
        super(BaseCanvas, self).__init__(*args, **kwargs)
        self._visuals = []

    def add_visual(self, visual):
        self._visuals.append(visual)
        visual.attach(self)

    def on_draw(self, e):
        gloo.clear()
        for visual in self._visuals:
            visual.draw()


class BaseVisual(object):
    _gl_primitive_type = None
    _shader_name = None

    def __init__(self):
        assert self._gl_primitive_type
        assert self._shader_name

        self.size = 1, 1
        self._canvas = None
        self._do_show = False

        self.program = _create_program(self._shader_name)

    def show(self):
        self._do_show = True

    def hide(self):
        self._do_show = False

    def set_data(self):
        """Set the data for the visual."""
        pass

    def set_transforms(self):
        """Set the list of transforms for the visual."""
        pass

    def attach(self, canvas):
        """Attach some events."""
        self._canvas = canvas

        @canvas.connect
        def on_resize(event):
            """Resize the OpenGL context."""
            self.size = event.size
            canvas.context.set_viewport(0, 0, event.size[0], event.size[1])

        @canvas.connect
        def on_mouse_move(event):
            if self._do_show:
                self.on_mouse_move(event)

    def on_mouse_move(self, e):
        pass

    def draw(self):
        """Draw the waveforms."""
        if self._do_show:
            self.program.draw(self._gl_primitive_type)

    def update(self):
        """Trigger a draw event in the canvas from the visual."""
        if self._canvas:
            self._canvas.update()
