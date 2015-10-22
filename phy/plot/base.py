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

class BaseVisual(object):
    """A Visual represents one object (or homogeneous set of objects).

    It is rendered with a single pass of a single gloo program with a single
    type of GL primitive.

    Derived classes must implement:

    * `gl_primitive_type`: `lines`, `points`, etc.
    * `vertex` and `fragment`, or `shader_name`: the GLSL code, or the name of
      the GLSL files to load from the `glsl/` subdirectory.
    `shader_name`
    * `data`: a dictionary acting as a proxy for the gloo Program.
      This is because the Program is built later, once the interact has been
      attached. The interact is responsible for the creation of the program,
      since it implements a part of the transform chain.
    * `transforms`: a list of `Transform` instances, which can act on the CPU
      or the GPU. The interact's transforms will be appended to that list
      when the visual is attached to the canvas.

    """
    gl_primitive_type = None
    vertex = None
    fragment = None
    shader_name = None  # Use this to load shaders from the glsl/ library.

    def __init__(self):
        if self.shader_name:
            self.vertex = _load_shader(self.shader_name + '.vert')
            self.fragment = _load_shader(self.shader_name + '.frag')
        assert self.vertex
        assert self.fragment
        assert self.gl_primitive_type

        self.size = 1, 1
        self._canvas = None
        self.program = None
        # Not taken into account when the program has not been built.
        self._do_show = True

        # To set in `set_data()`.
        self.data = {}  # Data to set on the program when possible.
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
        """Set the data for the visual.

        Derived classes can add data to the `self.data` dictionary and
        set transforms in the `self.transforms` list.

        """
        pass

    def attach(self, canvas, interact='BaseInteract'):
        """Attach the visual to a canvas.

        The interact's name can be specified. The interact's transforms
        will be appended to the visual's transforms.

        """
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
        def on_mouse_wheel(event):
            if self._do_show:
                self.on_mouse_wheel(event)

        @canvas.connect
        def on_mouse_move(event):
            if self._do_show:
                self.on_mouse_move(event)

        @canvas.connect
        def on_key_press(event):
            if self._do_show:
                self.on_key_press(event)

    def on_mouse_move(self, e):
        pass

    def on_mouse_wheel(self, e):
        pass

    def on_key_press(self, e):
        pass

    def build_program(self, transforms=None, vertex_decl='', frag_decl=''):
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
        if self.transform_chain:
            # Insert the interact's GLSL into the shaders.
            self.vertex, self.fragment = self.transform_chain.insert_glsl(
                self.vertex, self.fragment)
            # Insert shader declarations.
            self.vertex = vertex_decl + '\n' + self.vertex
            self.fragment = frag_decl + '\n' + self.fragment
        logger.log(5, "Vertex shader: \n%s", self.vertex)
        logger.log(5, "Fragment shader: \n%s", self.fragment)
        self.program = gloo.Program(self.vertex, self.fragment)

        if not self.transform_chain.transformed_var_name:
            logger.debug("No transformed variable has been found.")
        # Upload the data if necessary.
        self._upload_data()

    def _upload_data(self):
        """Upload pending data (attributes and uniforms) before drawing."""
        if not self.data:
            return

        # Get the name of the variable that needs to be transformed.
        # This variable (typically a_position) comes from the vertex shader
        # which contains the string `gl_Position = transform(the_name);`.
        var = self.transform_chain.transformed_var_name

        logger.log(5, "Upload program objects %s.",
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
            # Upload pending data.
            self._upload_data()
            # Finally, draw the program.
            self.program.draw(self.gl_primitive_type)

    def update(self):
        """Trigger a draw event in the canvas from the visual."""
        if self._canvas:
            self._canvas.update()


class BaseCanvas(Canvas):
    """A blank VisPy canvas with a custom event system that keeps the order."""
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

    * Define a list of `transforms`

    """
    transforms = None
    vertex_decl = ''
    frag_decl = ''

    def __init__(self):
        self._canvas = None
        self.data = {}

    @property
    def size(self):
        return self._canvas.size if self._canvas else None

    def attach(self, canvas):
        """Attach the interact to a canvas."""
        self._canvas = canvas

        @canvas.connect_
        def on_draw():
            # Build the programs of all attached visuals.
            # Programs that are already built are skipped.
            self.build_programs()

        canvas.connect(self.on_resize)
        canvas.connect(self.on_mouse_move)
        canvas.connect(self.on_mouse_wheel)
        canvas.connect(self.on_key_press)

    def is_attached(self):
        """Whether the transform is attached to a canvas."""
        return self._canvas is not None

    def iter_attached_visuals(self):
        """Yield all visuals attached to that interact in the canvas."""
        if self._canvas:
            for visual in self._canvas.emit_('get_visual_for_interact',
                                             self.__class__.__name__):
                if visual:
                    yield visual

    def build_programs(self):
        """Build the programs of all attached visuals.

        The list of transforms of the interact should have been set before
        calling this function.

        """
        for visual in self.iter_attached_visuals():
            if not visual.program:
                # Use the interact's data by default.
                for n, v in self.data.items():
                    if n not in visual.data:
                        visual.data[n] = v
                visual.build_program(self.transforms,
                                     vertex_decl=self.vertex_decl,
                                     frag_decl=self.frag_decl,
                                     )

    def on_resize(self, event):
        pass

    def on_mouse_move(self, event):
        pass

    def on_mouse_wheel(self, event):
        pass

    def on_key_press(self, event):
        pass

    def update(self):
        """Update the attached canvas if it exists."""
        if self.is_attached():
            self._canvas.update()
