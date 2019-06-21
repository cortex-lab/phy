"""Show how to create a custom matplotlib view in the GUI."""

from phy import IPlugin
from phy.cluster.views import ManualClusteringView  # Base class for phy views
from phy.plot.plot import PlotCanvasMpl  # matplotlib canvas


class FeatureDensityView(ManualClusteringView):
    plot_canvas_class = PlotCanvasMpl  # use matplotlib instead of OpenGL (the default)

    def __init__(self, features=None):
        """features is a function (cluster_id => Bunch(data, ...)) where data is a 3D array."""
        super(FeatureDensityView, self).__init__()
        self.features = features

    def on_select(self, cluster_ids=(), **kwargs):
        self.cluster_ids = cluster_ids
        # We don't display anything if no clusters are selected.
        if not cluster_ids:
            return

        # To simplify, we only consider the first PC component of the first 2 best channels.
        # Note that the features are in sparse format, where data's shape is
        # (n_spikes, n_best_channels, n_pcs). Only best channels for that clusters are
        # considered.
        # For this example, we just take the first 2 dimensions.
        x, y = self.features(cluster_ids[0]).data[:, :2, 0].T

        # We draw a 2D histogram with matplotlib.
        # The objects are:
        # - self.figure, a Figure instance
        # - self.canvas, a PlotCanvasMpl instance
        # - self.canvas.ax, an Axes object.
        self.canvas.ax.hist2d(x, y, 50)

        # Use this to update the matplotlib figure.
        self.canvas.update()


class ExampleMatplotlibViewPlugin(IPlugin):
    def attach_to_controller(self, controller):
        def create_feature_density_view():
            """A function that creates and returns a view."""
            return FeatureDensityView(features=controller._get_features)

        controller.view_creator['FeatureDensityView'] = create_feature_density_view
