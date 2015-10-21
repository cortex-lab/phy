# -*- coding: utf-8 -*-

"""Base VisPy classes."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging

from vispy import gloo
from vispy.app import Canvas

from .transform import TransformChain
from .utils import _load_shader
from phy.utils import EventEmitter

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Base spike visual
#------------------------------------------------------------------------------

def _build_program(name, transform_chain=None):
    vertex = _load_shader(name + '.vert')
    fragment = _load_shader(name + '.frag')

    if transform_chain:
        vertex, fragment = transform_chain.insert_glsl(vertex, fragment)

    program = gloo.Program(vertex, fragment)
    return program


class BaseVisual(object):
    gl_primitive_type = None
    shader_name = None

    def __init__(self):
        assert self.gl_primitive_type
        assert self.shader_name

        self.size = 1, 1
        self._canvas = None
        # Not taken into account when the program has not been built.
        self._do_show = True
        self.data = {}  # Data to set on the program when possible.
        self.program = None
        self.transforms = []
        # Combine the visual's transforms and the interact transforms.
        # The interact triggers the creation of the transform chain in
        # self.build_program().
        self.transform_chain = None

    def show(self):
        self._do_show = True

    def hide(self):
        self._do_show = False

    def set_data(self):
        """Set the data for the visual."""
        pass

    def attach(self, canvas, interact='base'):
        """Attach some events."""
        logger.debug("Attach `%s` with interact `%s` to canvas.",
                     self.__class__.__name__, interact or '')
        self._canvas = canvas

        # Used when the canvas requests all attached visuals
        # for the given interact.
        @canvas.connect_
        def on_get_visual_for_interact(interact_req):
            if interact_req == interact:
                return self

        # NOTE: this is connect_ and not connect because we're using
        # phy's event system, not VisPy's. The reason is that the order
        # of the callbacks is not kept by VisPy, whereas we need the order
        # to draw visuals in the order they are attached.
        @canvas.connect_
        def on_draw():
            self.draw()

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

    def build_program(self, transforms=None):
        """Create the gloo program by specifying the transforms
        given by the optionally-attached interact.

        This function also uploads all variables set in `self.data` in
        `self.set_data()`.

        This function is called by the interact's `build_programs()` method
        during the draw event (only effective the first time necessary).

        """
        transforms = transforms or []
        assert self.program is None, "The program has already been built."

        # Build the transform chain using the visuals transforms first,
        # and the interact's transforms then.
        self.transform_chain = TransformChain(self.transforms + transforms)

        logger.debug("Build the program of `%s`.", self.__class__.__name__)
        self.program = _build_program(self.shader_name, self.transform_chain)

        # Get the name of the variable that needs to be transformed.
        # This variable (typically a_position) comes from the vertex shader
        # which contains the string `gl_Position = transform(the_name);`.
        var = self.transform_chain.transformed_var_name
        if not var:
            logger.debug("No transformed variable has been found.")

        # Upload the data if necessary.
        logger.debug("Upload program objects %s.",
                     ', '.join(self.data.keys()))
        for name, value in self.data.items():
            # Normalize the value that needs to be transformed.
            if name == var:
                value = self.transform_chain.apply(value)
            self.program[name] = value
        self.data.clear()

    def draw(self):
        """Draw the visual."""
        # Skip the drawing if the program hasn't been built yet.
        # The program is built by the attached interact.
        if self._do_show and self.program:
            # Finally, draw the program.
            self.program.draw(self.gl_primitive_type)

    def update(self):
        """Trigger a draw event in the canvas from the visual."""
        if self._canvas:
            self._canvas.update()


class BaseCanvas(Canvas):
    def __init__(self, *args, **kwargs):
        super(BaseCanvas, self).__init__(*args, **kwargs)
        self._events = EventEmitter()

    def connect_(self, *args, **kwargs):
        return self._events.connect(*args, **kwargs)

    def emit_(self, *args, **kwargs):
        return self._events.emit(*args, **kwargs)

    def on_draw(self, e):
        gloo.clear()
        self._events.emit('draw')


class BaseInteract(object):
    """Implement interactions for a set of attached visuals in a canvas.

    Derived classes must:

    * Define a unique `self.name`
    * Define a list of transforms

    """
    name = 'base'
    transforms = None

    def __init__(self):
        self._canvas = None

    def attach(self, canvas):
        """Attach the interact to a canvas."""
        self._canvas = canvas

        @canvas.connect_
        def on_draw():
            # Build the programs of all attached visuals.
            # Programs that are already built are skipped.
            self.build_programs()

        canvas.connect(self.on_mouse_move)
        canvas.connect(self.on_key_press)

    def iter_attached_visuals(self):
        """Yield all visuals attached to that interact in the canvas."""
        for visual in self._canvas.emit_('get_visual_for_interact', self.name):
            if visual:
                yield visual

    def build_programs(self):
        """Build the programs of all attached visuals.

        The transform chain of the interact must have been built before.

        """
        for visual in self.iter_attached_visuals():
            if not visual.program:
                assert self.transforms
                visual.build_program(self.transforms)

    def on_mouse_move(self, event):
        pass

    def on_key_press(self, event):
        pass
