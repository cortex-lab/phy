"""Show how to add a custom cluster histogram view showing cluster statistics."""

from phy import IPlugin, Bunch
from phy.cluster.views import HistogramView


class FeatureHistogramView(HistogramView):
    """Every view corresponds to a unique view class, so we need to subclass HistogramView."""
    n_bins = 100  # default number of bins
    x_max = .1  # maximum value on the x axis (maximum bin)
    alias_char = 'fh'  # provide `:fhn` (set number of bins) and `:fhm` (set max bin) snippets


class ExampleClusterStatsPlugin(IPlugin):
    def attach_to_controller(self, controller):

        def feature_histogram(cluster_id):
            """Must return a Bunch object with data and optional x_max, plot, text items.

            The histogram is automatically computed by the view, this function should return
            the original data used to compute the histogram, rather than the histogram itself.

            """
            return Bunch(data=controller.get_features(cluster_id).data)

        def create_view():
            """Create and return a histogram view."""
            return FeatureHistogramView(cluster_stat=feature_histogram)

        # Maps a view name to a function that returns a view
        # when called with no argument.
        controller.view_creator['FeatureHistogram'] = create_view
