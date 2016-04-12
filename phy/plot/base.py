# -*- coding: utf-8 -*-

"""Base VisPy classes."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from collections import defaultdict
import logging
import re

from vispy import gloo
from vispy.app import Canvas
from vispy.util.event import Event

from .transform import TransformChain, Clip
from .utils import _load_shader, _enable_depth_mask

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

    """

    """Data variables that can be lists of arrays."""
    allow_list = ()

    def __init__(self):
        self.gl_primitive_type = None
        self.transforms = TransformChain()
        self.inserter = GLSLInserter()
        self.inserter.insert_vert('uniform vec2 u_window_size;', 'header')
        # The program will be set by the canvas when the visual is
        # added to the canvas.
        self.program = None
        self.set_canvas_transforms_filter(lambda t: t)

    # Visual definition
    # -------------------------------------------------------------------------

    def set_shader(self, name):
        self.vertex_shader = _load_shader(name + '.vert')
        self.fragment_shader = _load_shader(name + '.frag')

    def set_primitive_type(self, primitive_type):
        self.gl_primitive_type = primitive_type

    def on_draw(self):
        """Draw the visual."""
        # Skip the drawing if the program hasn't been built yet.
        # The program is built by the interact.
        if self.program:
            # Draw the program.
            self.program.draw(self.gl_primitive_type)
        else:  # pragma: no cover
            logger.debug("Skipping drawing visual `%s` because the program "
                         "has not been built yet.", self)

    def on_resize(self, size):
        # HACK: we check whether u_window_size is used in order to avoid
        # the VisPy warning. We only update it if that uniform is active.
        s = '\n'.join(self.program.shaders)
        s = s.replace('uniform vec2 u_window_size;', '')
        if 'u_window_size' in s:
            self.program['u_window_size'] = size

    # To override
    # -------------------------------------------------------------------------

    @staticmethod
    def validate(**kwargs):
        """Make consistent the input data for the visual."""
        return kwargs  # pragma: no cover

    @staticmethod
    def vertex_count(**kwargs):
        """Return the number of vertices as a function of the input data."""
        return 0  # pragma: no cover

    def set_data(self):
        """Set data to the program.

        Must be called *after* attach(canvas), because the program is built
        when the visual is attached to the canvas.

        """
        raise NotImplementedError()

    def set_canvas_transforms_filter(self, f):
        """Set a function filtering the canvas' transforms."""
        self.canvas_transforms_filter = f


#------------------------------------------------------------------------------
# Build program with interacts
#------------------------------------------------------------------------------

def _insert_glsl(vertex, fragment, to_insert):
    """Insert snippets in a shader.

    to_insert is a dict `{(shader_type, location): snippet}`.

    Snippets can contain `{{ var }}` placeholders for the transformed variable
    name.

    """
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
    assert var and var in vertex

    # Headers.
    vertex = to_insert['vert', 'header'] + '\n\n' + vertex
    fragment = to_insert['frag', 'header'] + '\n\n' + fragment

    # Get the pre and post transforms.
    vs_insert = to_insert['vert', 'before_transforms']
    vs_insert += to_insert['vert', 'transforms']
    vs_insert += to_insert['vert', 'after_transforms']

    # Insert the GLSL snippet in the vertex shader.
    vertex = vs_regex.sub(indent(vs_insert), vertex)

    # Now, we make the replacements in the fragment shader.
    fs_regex = re.compile(r'(void main\(\)\s*\{)')
    # NOTE: we add the `void main(){` that was removed by the regex.
    fs_insert = '\\1\n' + to_insert['frag', 'before_transforms']
    fragment = fs_regex.sub(indent(fs_insert), fragment)

    # Replace the transformed variable placeholder by its name.
    vertex = vertex.replace('{{ var }}', var)

    return vertex, fragment


class GLSLInserter(object):
    """Insert GLSL snippets into shader codes."""

    def __init__(self):
        self._to_insert = defaultdict(list)
        self.insert_vert('vec2 temp_pos_tr = {{ var }};',
                         'before_transforms')
        self.insert_vert('gl_Position = vec4(temp_pos_tr, 0., 1.);',
                         'after_transforms')
        self.insert_vert('varying vec2 v_temp_pos_tr;\n', 'header')
        self.insert_frag('varying vec2 v_temp_pos_tr;\n', 'header')

    def _insert(self, shader_type, glsl, location):
        assert location in (
            'header',
            'before_transforms',
            'transforms',
            'after_transforms',
        )
        self._to_insert[shader_type, location].append(glsl)

    def insert_vert(self, glsl, location='transforms'):
        """Insert a GLSL snippet into the vertex shader.

        The location can be:

        * `header`: declaration of GLSL variables
        * `before_transforms`: just before the transforms in the vertex shader
        * `transforms`: where the GPU transforms are applied in the vertex
          shader
        * `after_transforms`: just after the GPU transforms

        """
        self._insert('vert', glsl, location)

    def insert_frag(self, glsl, location=None):
        """Insert a GLSL snippet into the fragment shader."""
        self._insert('frag', glsl, location)

    def add_transform_chain(self, tc):
        """Insert the GLSL snippets of a transform chain."""
        # Generate the transforms snippet.
        for t in tc.gpu_transforms:
            if isinstance(t, Clip):
                # Set the varying value in the vertex shader.
                self.insert_vert('v_temp_pos_tr = temp_pos_tr;')
                continue
            self.insert_vert(t.glsl('temp_pos_tr'))
        # Clipping.
        clip = tc.get('Clip')
        if clip:
            self.insert_frag(clip.glsl('v_temp_pos_tr'), 'before_transforms')

    def insert_into_shaders(self, vertex, fragment):
        """Apply the insertions to shader code."""
        to_insert = defaultdict(str)
        to_insert.update({key: '\n'.join(self._to_insert[key]) + '\n'
                          for key in self._to_insert})
        return _insert_glsl(vertex, fragment, to_insert)

    def __add__(self, inserter):
        """Concatenate two inserters."""
        for key, values in self._to_insert.items():
            values.extend([_ for _ in inserter._to_insert[key]
                           if _ not in values])
        return self


#------------------------------------------------------------------------------
# Base canvas
#------------------------------------------------------------------------------

class VisualEvent(Event):
    def __init__(self, type, visual=None):
        super(VisualEvent, self).__init__(type)
        self.visual = visual


class BaseCanvas(Canvas):
    """A blank VisPy canvas with a custom event system that keeps the order."""
    def __init__(self, *args, **kwargs):
        super(BaseCanvas, self).__init__(*args, **kwargs)
        self.transforms = TransformChain()
        self.inserter = GLSLInserter()
        self.visuals = []
        self.events.add(visual_added=VisualEvent)

        # Enable transparency.
        _enable_depth_mask()

    def add_visual(self, visual):
        """Add a visual to the canvas, and build its program by the same
        occasion.

        We can't build the visual's program before, because we need the canvas'
        transforms first.

        """
        # Retrieve the visual's GLSL inserter.
        inserter = visual.inserter
        # Add the visual's transforms.
        inserter.add_transform_chain(visual.transforms)
        # Then, add the canvas' transforms.
        canvas_transforms = visual.canvas_transforms_filter(self.transforms)
        inserter.add_transform_chain(canvas_transforms)
        # Also, add the canvas' inserter.
        inserter += self.inserter
        # Now, we insert the transforms GLSL into the shaders.
        vs, fs = visual.vertex_shader, visual.fragment_shader
        vs, fs = inserter.insert_into_shaders(vs, fs)
        # Finally, we create the visual's program.
        visual.program = gloo.Program(vs, fs)
        logger.log(5, "Vertex shader: %s", vs)
        logger.log(5, "Fragment shader: %s", fs)
        # Initialize the size.
        visual.on_resize(self.size)
        # Register the visual in the list of visuals in the canvas.
        self.visuals.append(visual)
        self.events.visual_added(visual=visual)

    def on_resize(self, event):
        """Resize the OpenGL context."""
        self.context.set_viewport(0, 0, event.size[0], event.size[1])
        for visual in self.visuals:
            visual.on_resize(event.size)
        self.update()

    def on_draw(self, e):
        """Draw all visuals."""
        gloo.clear()
        for visual in self.visuals:
            logger.log(5, "Draw visual `%s`.", visual)
            visual.on_draw()


#------------------------------------------------------------------------------
# Base interact
#------------------------------------------------------------------------------

class BaseInteract(object):
    """Implement dynamic transforms on a canvas."""
    canvas = None

    def attach(self, canvas):
        """Attach this interact to a canvas."""
        self.canvas = canvas

        @canvas.connect
        def on_visual_added(e):
            self.update_program(e.visual.program)

    def update_program(self, program):
        """Override this method to update programs when `self.update()`
        is called."""
        pass

    def update(self):
        """Update all visuals in the attached canvas."""
        if not self.canvas:
            return
        for visual in self.canvas.visuals:
            self.update_program(visual.program)
        self.canvas.update()
