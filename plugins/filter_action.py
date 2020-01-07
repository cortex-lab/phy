"""Show how to create a filter snippet for the cluster view.

Typing `:fr 10` automatically shows only the clusters that have a firing rate higher than 10 spk/s.

"""

from phy import IPlugin, connect


class ExampleFilterFiringRatePlugin(IPlugin):
    def attach_to_controller(self, controller):
        @connect
        def on_gui_ready(sender, gui):
            @gui.view_actions.add(alias='fr')  # corresponds to `:fr` snippet
            def filter_firing_rate(rate):
                """Filter clusters with the firing rate."""
                controller.supervisor.filter('fr > %.1f' % float(rate))
