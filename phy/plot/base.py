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

from .transform import TransformChain, Clip
from .utils import _load_shader

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
    def __init__(self):
        self.gl_primitive_type = None
        self.transforms = TransformChain()
        self.inserter = GLSLInserter()
        # The program will be set by the canvas when the visual is
        # added to the canvas.
        self.program = None

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

    # To override
    # -------------------------------------------------------------------------

    def set_data(self):
        """Set data to the program.

        Must be called *after* attach(canvas), because the program is built
        when the visual is attached to the canvas.

        """
        raise NotImplementedError()


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
        self._insert('vert', glsl, location)

    def insert_frag(self, glsl, location=None):
        self._insert('frag', glsl, location)

    def add_transform_chain(self, tc):
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
        to_insert = defaultdict(str)
        to_insert.update({key: '\n'.join(self._to_insert[key])
                          for key in self._to_insert})
        return _insert_glsl(vertex, fragment, to_insert)


#------------------------------------------------------------------------------
# Base canvas
#------------------------------------------------------------------------------

class BaseCanvas(Canvas):
    """A blank VisPy canvas with a custom event system that keeps the order."""
    def __init__(self, *args, **kwargs):
        super(BaseCanvas, self).__init__(*args, **kwargs)
        self.transforms = TransformChain()
        self.visuals = []

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
        inserter.add_transform_chain(self.transforms)
        # Now, we insert the transforms GLSL into the shaders.
        vs, fs = visual.vertex_shader, visual.fragment_shader
        vs, fs = inserter.insert_into_shaders(vs, fs)
        # Finally, we create the visual's program.
        visual.program = gloo.Program(vs, fs)
        logger.log(5, "Vertex shader: %s", vs)
        logger.log(5, "Fragment shader: %s", fs)
        # Register the visual in the list of visuals in the canvas.
        self.visuals.append(visual)

    def on_resize(self, event):
        """Resize the OpenGL context."""
        self.context.set_viewport(0, 0, event.size[0], event.size[1])

    def on_draw(self, e):
        gloo.clear()
        for visual in self.visuals:
            visual.on_draw()
