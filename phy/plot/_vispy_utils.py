# -*- coding: utf-8 -*-

"""Plotting/VisPy utilities."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op

import numpy as np

from vispy import app, gloo, config
from vispy.visuals import Visual
from vispy.visuals.shaders import ModularProgram

from ..utils.array import _unique, _as_array
from ..utils.logging import debug


#------------------------------------------------------------------------------
# Misc
#------------------------------------------------------------------------------

def _load_shader(filename):
    """Load a shader file."""
    path = op.join(op.dirname(op.realpath(__file__)), 'glsl', filename)
    with open(path, 'r') as f:
        return f.read()


def _tesselate_histogram(hist):
    assert hist.ndim == 1
    nsamples = len(hist)
    dx = 2. / nsamples

    x0 = -1 + dx * np.arange(nsamples)

    x = np.zeros(5 * nsamples + 1)
    y = -1.0 * np.ones(5 * nsamples + 1)

    x[0:-1:5] = x0
    x[1::5] = x0
    x[2::5] = x0 + dx
    x[3::5] = x0
    x[4::5] = x0 + dx
    x[-1] = 1

    y[1::5] = y[2::5] = -1 + 2. * hist

    return np.c_[x, y]


#------------------------------------------------------------------------------
# PanZoom class
#------------------------------------------------------------------------------

class PanZoom(object):

    """
    Pan & Zoom transform

    The panzoom transform allow to translate and scale an object in the window
    space coordinate (2D). This means that whatever point you grab on the
    screen, it should remains under the mouse pointer. Zoom is realized using
    the mouse scroll and is always centered on the mouse pointer.

    The transform is connected to the following events:

    * resize (update)
    * mouse_scroll (zoom)
    * mouse_grab (pan)

    You can also control programatically the transform using:

    * aspect: control the aspect ratio of the whole scene
    * pan   : translate the scene to the given 2D coordinates
    * zoom  : set the zoom level (centered at current pan coordinates)
    * zmin  : minimum zoom level
    * zmax  : maximum zoom level
    """

    def __init__(self, aspect=1.0, pan=(0.0, 0.0), zoom=1.0,
                 zmin=0.01, zmax=100.0):
        """
        Initialize the transform.

        Parameters
        ----------

        aspect : float (default is None)
           Indicate what is the aspect ratio of the object displayed. This is
           necessary to convert pixel drag move in oject space coordinates.

        pan : float, float (default is 0,0)
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
    def is_attached(self):
        """ Whether transform is attached to a canvas """

        return self._canvas is not None

    @property
    def aspect(self):
        """ Aspect (width/height) """

        return self._aspect

    @aspect.setter
    def aspect(self, value):
        """ Aspect (width/height) """

        self._aspect = value

    @property
    def pan(self):
        """ Pan translation """

        return self._pan

    @pan.setter
    def pan(self, value):
        """ Pan translation """

        self._pan = np.asarray(value)
        self._u_pan = self._pan
        for program in self._programs:
            program["u_pan"] = self._u_pan

    @property
    def zoom(self):
        """ Zoom level """

        return self._zoom

    @zoom.setter
    def zoom(self, value):
        """ Zoom level """

        self._zoom = max(min(value, self._zmax), self._zmin)
        if not self.is_attached:
            return

        aspect = 1.0
        if self._aspect is not None:
            aspect = self._canvas_aspect * self._aspect

        self._u_zoom = self._zoom * aspect
        for program in self._programs:
            program["u_zoom"] = self._u_zoom

    @property
    def zmin(self):
        """ Minimum zoom level """

        return self._zmin

    @zmin.setter
    def zmin(self, value):
        """ Minimum zoom level """

        self._zmin = min(value, self._zmax)

    @property
    def zmax(self):
        """ Maximal zoom level """

        return self._zmax

    @zmax.setter
    def zmax(self, value):
        """ Maximal zoom level """

        self._zmax = max(value, self._zmin)

    def on_resize(self, event):
        """ Resize event """

        self._width = float(event.size[0])
        self._height = float(event.size[1])
        aspect = self._width / self._height
        if aspect > 1.0:
            self._canvas_aspect = np.array([1.0 / aspect, 1.0])
        else:
            self._canvas_aspect = np.array([1.0, aspect / 1.0])

        # Update zoom level
        self.zoom = self._zoom

    def on_mouse_move(self, event):
        """ Drag """

        if not event.is_dragging:
            return

        x, y = event.pos
        dx = +2 * ((x - event.last_event.pos[0]) / self._width)
        dy = -2 * ((y - event.last_event.pos[1]) / self._height)

        self.pan += dx, dy
        self._canvas.update()

    def on_mouse_wheel(self, event):
        """ Zoom """

        x, y = event.pos
        dx, dy = event.delta
        dx /= 10.0
        dy /= 10.0

        # Normalize mouse coordinates and invert y axis
        x = x / (self._width / 2.) - 1.0
        y = 1.0 - y / (self._height / 2.0)

        zoom = min(max(self._zoom * (1.0 + dy), self._zmin), self._zmax)
        # ratio = zoom / self.zoom
        # xpan = x - ratio * (x - self.pan[0])
        # ypan = y - ratio * (y - self.pan[1])
        self.zoom = zoom
        # self.pan = xpan, ypan
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

        aspect = 1.0
        if self._aspect is not None:
            aspect = self._canvas_aspect * self._aspect
        self._u_zoom = self._zoom * aspect

        canvas.connect(self.on_resize)
        canvas.connect(self.on_mouse_wheel)
        canvas.connect(self.on_mouse_move)


#------------------------------------------------------------------------------
# Base spike visual
#------------------------------------------------------------------------------

class BaseSpikeVisual(Visual):
    _shader_name = ''
    _gl_draw_mode = ''

    def __init__(self, **kwargs):
        super(BaseSpikeVisual, self).__init__(**kwargs)
        self.n_spikes = None
        self._spike_clusters = None
        self._spike_ids = None
        self._non_empty = False
        self._to_bake = []

        vertex = _load_shader(self._shader_name + '.vert')
        fragment = _load_shader(self._shader_name + '.frag')

        curdir = op.dirname(op.realpath(__file__))
        config['include_path'] = [op.join(curdir, 'glsl')]

        self.program = ModularProgram(vertex, fragment)

        gloo.set_state(clear_color='black', blend=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))

    # Data properties
    # -------------------------------------------------------------------------

    def _set_or_assert_n_spikes(self, arr):
        """If n_spikes is None, set it using the array's shape. Otherwise,
        check that the array has n_spikes rows."""
        if self.n_spikes is None:
            self.n_spikes = arr.shape[0]
        assert arr.shape[0] == self.n_spikes

    def set_to_bake(self, *bakes):
        for bake in bakes:
            if bake not in self._to_bake:
                self._to_bake.append(bake)

    @property
    def spike_clusters(self):
        """The clusters assigned to *all* spikes, not just the displayed
        spikes."""
        return self._spike_clusters

    @spike_clusters.setter
    def spike_clusters(self, value):
        """Set all spike clusters."""
        value = _as_array(value)
        self._spike_clusters = value
        self.set_to_bake('spikes_clusters')

    @property
    def masks(self):
        """Masks of the displayed waveforms."""
        return self._masks

    @masks.setter
    def masks(self, value):
        value = _as_array(value)
        self._set_or_assert_n_spikes(value)
        # TODO: support sparse structures
        assert value.ndim == 2
        assert value.shape == (self.n_spikes, self.n_channels)
        self._masks = value
        self.set_to_bake('spikes')

    @property
    def spike_ids(self):
        """The list of spike ids to display, should correspond to the
        waveforms."""
        if self._spike_ids is None:
            self._spike_ids = np.arange(self.n_spikes).astype(np.int64)
        return self._spike_ids

    @spike_ids.setter
    def spike_ids(self, value):
        value = _as_array(value)
        self._set_or_assert_n_spikes(value)
        self._spike_ids = value
        self.set_to_bake('spikes')

    # TODO: channel_ids

    @property
    def cluster_ids(self):
        """Clusters of the displayed spikes."""
        return _unique(self.spike_clusters[self.spike_ids])

    @property
    def n_clusters(self):
        return len(self.cluster_ids)

    @property
    def cluster_colors(self):
        """Colors of the displayed clusters."""
        return self._cluster_colors

    @cluster_colors.setter
    def cluster_colors(self, value):
        self._cluster_colors = _as_array(value)
        assert len(self._cluster_colors) == self.n_clusters
        self.set_to_bake('color')

    # Data baking
    # -------------------------------------------------------------------------

    def _bake_color(self):
        u_cluster_color = self.cluster_colors.reshape((1, self.n_clusters, -1))
        u_cluster_color = (u_cluster_color * 255).astype(np.uint8)
        # TODO: more efficient to update the data from an existing texture
        self.program['u_cluster_color'] = gloo.Texture2D(u_cluster_color)
        debug("bake color", u_cluster_color.shape)

    def _bake(self):
        """Prepare and upload the data on the GPU.

        Return whether something has been baked or not.

        """
        if not self._non_empty:
            return
        n_bake = len(self._to_bake)
        # Bake what needs to be baked.
        # WARNING: the bake functions are called in alphabetical order.
        # Tweak the names if there are dependencies between the functions.
        for bake in sorted(self._to_bake):
            # Name of the private baking method.
            name = '_bake_{0:s}'.format(bake)
            if hasattr(self, name):
                getattr(self, name)()
        self._to_bake = []
        return n_bake > 0

    def draw(self):
        """Draw the waveforms."""
        # Bake what needs to be baked at this point.
        self._bake()
        if self._non_empty:
            self.program.draw(self._gl_draw_mode)


#------------------------------------------------------------------------------
# Base spike canvas
#------------------------------------------------------------------------------

class BaseSpikeCanvas(app.Canvas):
    _visual_class = None

    def __init__(self, **kwargs):
        super(BaseSpikeCanvas, self).__init__(keys='interactive', **kwargs)
        self.visual = self._visual_class()
        self._pz = PanZoom()
        self._pz.add(self.visual.program)
        self._pz.attach(self)

    def on_draw(self, event):
        gloo.clear('black')
        self.visual.draw()

    def on_resize(self, event):
        gloo.set_viewport(0, 0, event.size[0], event.size[1])
