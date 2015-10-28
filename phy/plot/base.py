# -*- coding: utf-8 -*-

"""Base VisPy classes."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import re

from vispy import gloo
from vispy.app import Canvas

from .transform import TransformChain, GPU, Clip
from .utils import _load_shader
from phy.utils import EventEmitter

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------------

def indent(text):
    return '\n'.join('    ' + l.strip() for l in text.splitlines())


#------------------------------------------------------------------------------
# Base spike visual
#------------------------------------------------------------------------------

class BaseVisual(object):
    """A Visual represents one object (or homogeneous set of objects).

    It is rendered with a single pass of a single gloo program with a single
    type of GL primitive.

    Derived classes must implement:

    * `gl_primitive_type`: `lines`, `points`, etc.
    * `get_shaders()`: return the vertex and fragment shaders, or just
      `shader_name` for built-in shaders
    * `get_transforms()`: return a list of `Transform` instances, which
    * `get_post_transforms()`: return a GLSL snippet to insert after
      all transforms in the vertex shader.
    * `set_data()`: has access to `self.program`. Must be called after
      `attach()`.

    """
    gl_primitive_type = None
    shader_name = None

    def __init__(self):
        # This will be set by attach().
        self.program = None

    # To override
    # -------------------------------------------------------------------------

    def get_shaders(self):
        """Return the vertex and fragment shader code."""
        assert self.shader_name
        return (_load_shader(self.shader_name + '.vert'),
                _load_shader(self.shader_name + '.frag'))

    def get_transforms(self):
        """Return the list of transforms for the visual.

        There needs to be one and exactly one instance of `GPU()`.

        """
        return [GPU()]

    def get_post_transforms(self):
        """Return a GLSL snippet to insert after all transforms in the
        vertex shader."""
        return ''

    def set_data(self):
        """Set data to the program.

        Must be called *after* attach(canvas), because the program is built
        when the visual is attached to the canvas.

        """
        raise NotImplementedError()

    # Public methods
    # -------------------------------------------------------------------------

    def apply_cpu_transforms(self, data):
        return TransformChain(self.get_transforms()).apply(data)

    def attach(self, canvas):
        """Attach the visual to a canvas.

        After calling this method, the following properties are available:

        * self.program

        """
        logger.debug("Attach `%s` to canvas.", self.__class__.__name__)

        self.program = build_program(self, canvas.interacts)

        # NOTE: this is connect_ and not connect because we're using
        # phy's event system, not VisPy's. The reason is that the order
        # of the callbacks is not kept by VisPy, whereas we need the order
        # to draw visuals in the order they are attached.
        @canvas.connect_
        def on_draw():
            self.on_draw()

        @canvas.connect
        def on_resize(event):
            """Resize the OpenGL context."""
            canvas.context.set_viewport(0, 0, event.size[0], event.size[1])

        canvas.connect(self.on_mouse_wheel)
        canvas.connect(self.on_mouse_move)
        canvas.connect(self.on_key_press)

        # NOTE: this might be improved.
        canvas.visuals.append(self)
        # HACK: allow a visual to update the canvas it is attached to.
        self.update = canvas.update

    def on_mouse_move(self, e):
        pass

    def on_mouse_wheel(self, e):
        pass

    def on_key_press(self, e):
        pass

    def on_draw(self):
        """Draw the visual."""
        # Skip the drawing if the program hasn't been built yet.
        # The program is built by the interact.
        if self.program:
            # Draw the program.
            self.program.draw(self.gl_primitive_type)


#------------------------------------------------------------------------------
# Base interact
#------------------------------------------------------------------------------

class BaseInteract(object):
    """Implement interactions for a set of attached visuals in a canvas.

    Derived classes must:

    * Define a list of `transforms`

    """
    def __init__(self):
        self._canvas = None

    # To override
    # -------------------------------------------------------------------------

    def get_shader_declarations(self):
        """Return extra declarations for the vertex and fragment shaders."""
        return '', ''

    def get_pre_transforms(self):
        """Return an optional GLSL snippet to insert into the vertex shader
        before the transforms."""
        return ''

    def get_transforms(self):
        """Return the list of transforms."""
        return []

    def update_program(self, program):
        """Update a program during an interaction event."""
        pass

    # Public methods
    # -------------------------------------------------------------------------

    @property
    def size(self):
        return self._canvas.size if self._canvas else None

    def attach(self, canvas):
        """Attach the interact to a canvas."""
        self._canvas = canvas

        # NOTE: this might be improved.
        canvas.interacts.append(self)

        canvas.connect(self.on_resize)
        canvas.connect(self.on_mouse_move)
        canvas.connect(self.on_mouse_wheel)
        canvas.connect(self.on_key_press)

    def is_attached(self):
        """Whether the interact is attached to a canvas."""
        return self._canvas is not None

    def on_resize(self, event):
        pass

    def on_mouse_move(self, event):
        pass

    def on_mouse_wheel(self, event):
        pass

    def on_key_press(self, event):
        pass

    def update(self):
        """Update the attached canvas and all attached programs."""
        if self.is_attached():
            for visual in self._canvas.visuals:
                self.update_program(visual.program)
            self._canvas.update()


#------------------------------------------------------------------------------
# Base canvas
#------------------------------------------------------------------------------

class BaseCanvas(Canvas):
    """A blank VisPy canvas with a custom event system that keeps the order."""
    def __init__(self, *args, **kwargs):
        kwargs['keys'] = 'interactive'
        super(BaseCanvas, self).__init__(*args, **kwargs)
        self._events = EventEmitter()
        self.interacts = []
        self.visuals = []

    def connect_(self, *args, **kwargs):
        return self._events.connect(*args, **kwargs)

    def emit_(self, *args, **kwargs):  # pragma: no cover
        return self._events.emit(*args, **kwargs)

    def on_draw(self, e):
        gloo.clear()
        self._events.emit('draw')


#------------------------------------------------------------------------------
# Build program with interacts
#------------------------------------------------------------------------------

def insert_glsl(transform_chain, vertex, fragment,
                pre_transforms='', post_transforms=''):
    """Generate the GLSL code of the transform chain."""

    # TODO: move this to base.py

    # Find the place where to insert the GLSL snippet.
    # This is "gl_Position = transform(data_var_name);" where
    # data_var_name is typically an attribute.
    vs_regex = re.compile(r'gl_Position = transform\(([\S]+)\);')
    r = vs_regex.search(vertex)
    if not r:
        logger.debug("The vertex shader doesn't contain the transform "
                     "placeholder: skipping the transform chain "
                     "GLSL insertion.")
        return vertex, fragment
    assert r
    logger.log(5, "Found transform placeholder in vertex code: `%s`",
               r.group(0))

    # Find the GLSL variable with the data (should be a `vec2`).
    var = r.group(1)
    transform_chain.transformed_var_name = var
    assert var and var in vertex

    # Generate the snippet to insert in the shaders.
    temp_var = 'temp_pos_tr'
    # Name for the (eventual) varying.
    fvar = 'v_{}'.format(temp_var)
    vs_insert = ''
    # Insert the pre-transforms.
    vs_insert += pre_transforms + '\n'
    vs_insert += "vec2 {} = {};\n".format(temp_var, var)
    for t in transform_chain.gpu_transforms:
        if isinstance(t, Clip):
            # Set the varying value in the vertex shader.
            vs_insert += '{} = {};\n'.format(fvar, temp_var)
            continue
        vs_insert += t.glsl(temp_var) + '\n'
    vs_insert += 'gl_Position = vec4({}, 0., 1.);\n'.format(temp_var)
    vs_insert += post_transforms + '\n'

    # Clipping.
    clip = transform_chain.get('Clip')
    if clip:
        # Varying name.
        glsl_clip = clip.glsl(fvar)

        # Prepare the fragment regex.
        fs_regex = re.compile(r'(void main\(\)\s*\{)')
        fs_insert = '\\1\n{}'.format(glsl_clip)

        # Add the varying declaration for clipping.
        varying_decl = 'varying vec2 {};\n'.format(fvar)
        vertex = varying_decl + vertex
        fragment = varying_decl + fragment

        # Make the replacement in the fragment shader for clipping.
        fragment = fs_regex.sub(indent(fs_insert), fragment)

    # Insert the GLSL snippet of the transform chain in the vertex shader.
    vertex = vs_regex.sub(indent(vs_insert), vertex)

    return vertex, fragment


def build_program(visual, interacts=()):
    """Create the gloo program of a visual using the interacts
    transforms.

    This method is called when a visual is attached to the canvas.

    """
    assert visual.program is None, "The program has already been built."

    # Build the transform chain using the visuals transforms first,
    # then the interact's transforms.
    transforms = visual.get_transforms()
    for interact in interacts:
        transforms.extend(interact.get_transforms())
    transform_chain = TransformChain(transforms)

    logger.debug("Build the program of `%s`.", visual.__class__.__name__)
    # Insert the interact's GLSL into the shaders.
    vertex, fragment = visual.get_shaders()
    # Get the GLSL snippet to insert before the transformations.
    pre = '\n'.join(interact.get_pre_transforms() for interact in interacts)
    # GLSL snippet to insert after all transformations.
    post = visual.get_post_transforms()
    vertex, fragment = insert_glsl(transform_chain, vertex, fragment,
                                   pre, post)

    # Insert shader declarations using the interacts (if any).
    if interacts:
        vertex_decls, frag_decls = zip(*(interact.get_shader_declarations()
                                         for interact in interacts))

        vertex = '\n'.join(vertex_decls) + '\n' + vertex
        fragment = '\n'.join(frag_decls) + '\n' + fragment

    logger.log(5, "Vertex shader: \n%s", vertex)
    logger.log(5, "Fragment shader: \n%s", fragment)

    program = gloo.Program(vertex, fragment)

    # Update the program with all interacts.
    for interact in interacts:
        interact.update_program(program)

    return program
