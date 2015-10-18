# -*- coding: utf-8 -*-

"""Base VisPy classes."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging

from .utils import _create_program

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Base spike visual
#------------------------------------------------------------------------------

class BaseVisual(object):
    _gl_primitive_type = None
    _shader_name = None

    def __init__(self):
        assert self._gl_primitive_type
        assert self._shader_name

        self._data = {'a_position': None}
        self._to_upload = []  # list of arrays/params to upload

        self.program = _create_program(self._shader_name)

    def is_empty(self):
        """Return whether the visual is empty."""
        return self._data['a_position'] is not None

    def set_data(self):
        pass

    def set_transforms(self):
        pass

    def attach(self, canvas):
        canvas.connect(self.on_draw)

        @canvas.connect
        def on_resize(event):
            """Resize the OpenGL context."""
            canvas.context.set_viewport(0, 0, event.size[0], event.size[1])

    def on_draw(self, e):
        """Draw the waveforms."""
        # Upload to the GPU what needs to be uploaded.
        for name in self._to_upload:
            value = self._data[name]
            logger.debug("Upload `%s`: %s.", name, str(value))
            self.program[name] = value
        # Reset the list of objects to upload.
        self._to_upload = []
        if not self.is_empty():
            self.program.draw(self._gl_primitive_type)
