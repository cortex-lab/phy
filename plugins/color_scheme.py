"""Show how to add a custom color scheme to a view."""

from phy import IPlugin, connect
from phy.cluster.views import ClusterScatterView


class ExampleColorSchemePlugin(IPlugin):
    def attach_to_controller(self, controller):
        # Initial actions when creating views.
        @connect
        def on_view_attached(view, gui):
            # We need the initial list of cluster ids to initialize the color map.
            cluster_ids = controller.supervisor.clustering.cluster_ids

            if isinstance(view, ClusterScatterView):
                # Each view has a set of color schemes among which one can cycle through in
                # the GUI.
                view.add_color_scheme(
                    name='mycolorscheme',
                    fun=controller.get_cluster_amplitude,  # cluster_id => value
                    colormap='rainbow',  # or use a colorcet color map or a custom N*3 array
                    cluster_ids=cluster_ids,
                )
