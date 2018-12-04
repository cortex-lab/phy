# -----------------------------------------------------------------------------
# Copyright (c) 2009-2016 Nicolas P. Rougier. All rights reserved.
# Distributed under the (new) BSD License.
# -----------------------------------------------------------------------------

from OpenGL.GL import *  # noqa
from OpenGL.GL.ARB.texture_rg import *  # noqa
from OpenGL.GL.NV.geometry_program4 import *  # noqa
from OpenGL.GL.EXT.geometry_shader4 import *  # noqa
from OpenGL import contextdata
from OpenGL.plugins import FormatHandler
import ctypes

import OpenGL
OpenGL.ERROR_ON_COPY = True
# -> if set to a True value before importing the numpy/lists support modules,
#    will cause array operations to raise OpenGL.error.CopyError if the
#    operation would cause a data-copy in order to make the passed data-type
#    match the target data-type.

FormatHandler('gloo',
              'OpenGL.arrays.numpymodule.NumpyHandler', [
                  'gloo.buffer.VertexBuffer',
                  'gloo.buffer.IndexBuffer',
                  'gloo.atlas.Atlas',
                  'gloo.texture.Texture2D',
                  'gloo.texture.Texture1D',
                  'gloo.texture.FloatTexture2D',
                  'gloo.texture.FloatTexture1D',
                  'gloo.texture.TextureCube',
              ])


def cleanupCallback(context=None):
    """Create a cleanup callback to clear context-specific storage for the current context"""
    def callback(context=contextdata.getContext(context)):
        """Clean up the context, assumes that the context will *not* render again!"""
        contextdata.cleanupContext(context)
    return callback


def clear(color=(0, 0, 0, 10)):
    glClearColor(*color)  # noqa
    glClear(GL_COLOR_BUFFER_BIT)  # noqa


def enable_depth_mask():
    glClearColor(0, 0, 0, 0)  # noqa
    glClearDepth(1.)  # noqa

    glEnable(GL_BLEND)  # noqa
    glDepthRange(0., 1.)  # noqa
    glDepthFunc(GL_EQUAL)  # noqa
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)  # noqa

    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE)  # noqa
    glEnable(GL_POINT_SPRITE)  # noqa


# Patch: pythonize the glGetActiveAttrib
_glGetActiveAttrib = glGetActiveAttrib  # noqa


def glGetActiveAttrib(program, index):
    # Prepare
    bufsize = 32
    length = ctypes.c_int()
    size = ctypes.c_int()
    type = ctypes.c_int()
    name = ctypes.create_string_buffer(bufsize)
    # Call
    _glGetActiveAttrib(program, index,
                       bufsize, ctypes.byref(length), ctypes.byref(size),
                       ctypes.byref(type), name)
    # Return Python objects
    return name.value, size.value, type.value


# # --- Wrapper ---
# import sys
# def wrap(name):
#     if callable(globals()[name]):
#         def wrapper(*args, **kwargs):
#             # print "Calling %s%s" % (name, args)
#             return globals()[name](*args, **kwargs)
#         return wrapper
#     else:
#         return globals()[name]
#
# class Wrapper(object):
#     def __init__(self, wrapped):
#         self.wrapped = wrapped
#     def __getattr__(self, name):
#         return wrap(name)
#
# sys.modules[__name__] = Wrapper(sys.modules[__name__])
