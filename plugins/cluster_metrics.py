"""Show how to add a custom cluster metrics."""

import numpy as np
from phy import IPlugin


class MyPlugin(IPlugin):
    def attach_to_controller(self, controller):
        """Note that this function is called at initialization time, *before* the supervisor is
        created. The `controller.cluster_metrics` items are then passed to the supervisor when
        constructing it."""

        def meanisi(cluster_id):
            t = controller.get_spike_times(cluster_id).data
            return np.diff(t).mean()

        # Use this dictionary to define custom cluster metrics.
        controller.cluster_metrics['meanisi'] = meanisi
