"""Show how to injet specific Python variables in the IPython view."""

from phy import IPlugin, connect
from phy.cluster.views import WaveformView
from phy.gui.widgets import IPythonView


class ExampleIPythonViewPlugin(IPlugin):
    def attach_to_controller(self, controller):
        @connect
        def on_view_attached(view, gui):
            # This is called whenever a new view is added to the GUI.
            if isinstance(view, IPythonView):
                # We inject the first WaveformView of the GUI to the IPython console.
                view.inject(wv=gui.get_view(WaveformView))

        # Open an IPython view if there is not already one.
        controller.at_least_one_view('IPythonView')
