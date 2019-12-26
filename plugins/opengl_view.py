"""Show how to write a custom OpenGL view. This is for advanced users only."""

import numpy as np

from phy.utils.color import selected_cluster_color

from phy import IPlugin
from phy.cluster.views import ManualClusteringView
from phy.plot.visuals import PlotVisual


class MyOpenGLView(ManualClusteringView):
    """All OpenGL views derive from ManualClusteringView."""

    def __init__(self, templates=None):
        """
        Typically, the constructor takes as arguments *functions* that take as input
        one or several cluster ids, and return as many Bunch instances which contain
        the data as NumPy arrays. Many such functions are defined in the TemplateController.
        """

        super(MyOpenGLView, self).__init__()

        """
        The View instance contains a special `canvas` object which is a `̀PlotCanvas` instance.
        This class derives from `BaseCanvas` which itself derives from the PyQt5 `QOpenGLWindow`.
        The canvas represents a rectangular black window where you can draw geometric objects
        with OpenGL.

        phy uses the notion of **Layout** that lets you organize graphical elements in different
        subplots. These subplots can be organized in several ways:

        * Grid layout: a `(n_rows, n_cols)` grid of subplots (example: FeatureView).
        * Boxed layout: boxes arbitrarily located (example: WaveformView, using the
          probe geometry)
        * Stacked layout: one column with `n_boxes` subplots (example: TraceView,
          one row per channel)

        In this example, we use the stacked layout, with one subplot per cluster. This number
        will change at each cluster selection, depending on the number of selected clusters.
        But initially, we just use 1 subplot.

        """
        self.canvas.set_layout('stacked', n_plots=1)

        self.templates = templates

        """
        phy uses the notion of **Visual**. This is a graphical element that is represented with
        a single type of graphical element. phy provides many visuals:

        * PlotVisual (plots)
        * ScatterVisual (points with a given marker type and different colors and sizes)
        * LineVisual (for lines segments)
        * HistogramVisual
        * PolygonVisual
        * TextVisual
        * ImageVisual

        Each visual comes with a single OpenGL program, which is defined by a vertex shader
        and a fragment shader. These are programs written in a C-like language called GLSL.
        A visual also comes with a primitive type, which can be points, line segments, or
        triangles. This is all a GPU is able to render, but the position and the color of
        these primitives can be entirely customized in the shaders.

        The vertex shader acts on data arrays represented as NumPy arrays.

        These low-level details are hidden by the visuals abstraction, so it is unlikely that
        you'll ever need to write your own visual.

        In ManualClusteringViews, you typically define one or several visuals. For example
        if you need to add text, you would add `self.text_visual = TextVisual()`.

        """
        self.visual = PlotVisual()

        """
        For internal reasons, you need to add all visuals (empty for now) directly to the
        canvas, in the view's constructor. Later, we will use the `visual.set_data()` method
        to update the visual's data and display something in the figure.

        """
        self.canvas.add_visual(self.visual)

    def on_select(self, cluster_ids=(), **kwargs):
        """
        The main method to implement in ManualClusteringView is `on_select()`, called whenever
        new clusters are selected.

        *Note*: `cluster_ids` contains the clusters selected in the cluster view, followed
        by clusters selected in the similarity view.

        """

        """
        This method should always start with these few lines of code.
        """
        self.cluster_ids = cluster_ids
        if not cluster_ids:
            return

        """
        We update the number of boxes in the stacked layout, which is the number of
        selected clusters.
        """
        self.canvas.stacked.n_boxes = len(cluster_ids)

        """
        We obtain the template data.
        """
        bunchs = {cluster_id: self.templates(cluster_id).data for cluster_id in cluster_ids}

        """
        For performance reasons, it is best to use as few visuals as possible. In this example,
        we want 1 waveform template per subplot. We will use a single visual covering all
        subplots at once. This is the key to achieve good performance with OpenGL in Python.
        However, this comes with the drawback that the programming interface is more complicated.

        In principle, we would have to concatenate all data (x and y coordinates) of all subplots
        to pass it to `self.visual.set_data()` in order to draw all subplots at once. But this
        is tedious.

        phy uses the notion of **batch**: for each subplot, we set *partial data* for the subplot
        which just prepares the data for concatenation *after* we're done with looping through
        all clusters. The concatenation happens in the special call
        `self.canvas.update_visual(self.visual)`.

        We need to call `visual.reset_batch()` before constructing a batch.

        """
        self.visual.reset_batch()

        """
        We iterate through all selected clusters.
        """
        for idx, cluster_id in enumerate(cluster_ids):
            bunch = bunchs[cluster_id]

            """
            In this example, we just keep the peak channel. Note that `bunch.template` is a
            2D array `(n_samples, n_channels)` where `n_channels` in the number of "best"
            channels for the cluster. The channels are sorted by decreasing template amplitude,
            so the first one is the peak channel. The channel ids can be found in
            `bunch.channel_ids`.
            """
            y = bunch.template[:, 0]

            """
            We decide to use, on the x axis, values ranging from -1 to 1. This is the
            standard viewport in OpenGL and phy.
            """
            x = np.linspace(-1., 1., len(y))

            """
            phy requires you to specify explicitly the x and y range of the plots.
            The `data_bounds` variable is a `(xmin, ymin, xmax, ymax)` tuple representing the
            lower-left and upper-right corners of a rectangle. By default, the data bounds
            of the entire view is (-1, -1, 1, 1), also called normalized device coordinates.
            Eventually, OpenGL uses this coordinate system for display, but phy provides
            a transform system to convert from different coordinate systems, both on the CPU
            and the GPU.

            Here, the x range is (-1, 1), and the y range is (m, M) where m and M are
            respectively the min and max of the template.
            """
            m, M = y.min(), y.max()
            data_bounds = (-1, m, +1, M)

            """
            This function gives the color of the i-th selected cluster. This is a 4-tuple with
            values between 0 and 1 for RGBA: red, green, blue, alpha channel (transparency,
            1 by default).
            """
            color = selected_cluster_color(idx)

            """
            The plot visual takes as input the x and y coordinates of the points, the color,
            and the data bounds.
            There is also a special keyword argument `box_index` which is the subplot index.
            In the stacked layout, this is just an integer identifying the subplot index, from
            top to bottom. Note that in the grid view, the box index is a pair (row, col).
            """
            self.visual.add_batch_data(
                x=x, y=y, color=color, data_bounds=data_bounds, box_index=idx)

        """
        After the loop, this special call automatically builds the data to upload to the GPU
        by concatenating the partial data set in `add_batch_data()`.
        """
        self.canvas.update_visual(self.visual)

        """
        After updating the data on the GPU, we need to refresh the canvas.
        """
        self.canvas.update()


class ExampleOpenGLViewPlugin(IPlugin):
    def attach_to_controller(self, controller):
        def create_my_view():
            return MyOpenGLView(templates=controller._get_template_waveforms)

        controller.view_creator['MyOpenGLView'] = create_my_view

        # Open a view if there is not already one.
        controller.at_least_one_view('MyOpenGLView')
