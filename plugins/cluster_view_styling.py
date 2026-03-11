"""Show how to customize the cluster view with Qt stylesheet fragments."""

from phy import IPlugin
from phy.cluster.supervisor import ClusterView


class ExampleClusterViewStylingPlugin(IPlugin):
    def attach_to_controller(self, controller):
        # We add a custom stylesheet fragment to the ClusterView.
        ClusterView._styles += """
            QHeaderView::section {
                color: #f5c542;
                font-weight: bold;
            }
        """
