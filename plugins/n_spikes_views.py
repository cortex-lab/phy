"""Show how to increase the number of spikes used by several views."""

from phy import IPlugin, connect


class ExampleNspikesViewsPlugin(IPlugin):
    def attach_to_controller(self, controller):
        # Number of "best" channels kept for displaying the waveforms.
        controller.model.n_closest_channels = 12

        # The best channels are selected among the N closest to the best (peak) channel if their
        # mean amplitude is greater than this fraction of the peak amplitude on the best channel.
        # If zero, just the N closest channels are kept as the best channels.
        controller.model.amplitude_threshold = 0

        # GUI state is restored after plugins attach, so set persisted controller
        # values in gui_ready if the plugin should win on every launch.
        @connect(sender=controller)
        def on_gui_ready(sender, gui):
            """Keep only the settings below that you want this plugin to own."""
            controller.n_spikes_waveforms = 250  # default: 100 per cluster
            controller.n_spikes_features = 5000  # default: 2,500 per cluster
            controller.n_spikes_features_background = 5000  # default: 2,500 total
            controller.n_spikes_amplitudes = 20000  # default: 10,000 per cluster
            controller.n_spikes_correlograms = 250000  # default: 100,000 per cluster
