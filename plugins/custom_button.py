"""Show how to add custom buttons in a view's title bar."""

from phy import IPlugin, connect
from phy.cluster.views import WaveformView


class ExampleCustomButtonPlugin(IPlugin):
    def attach_to_controller(self, controller):
        @connect
        def on_view_attached(view, gui):
            if isinstance(view, WaveformView):

                # view.dock is a DockWidget instance, it has methods such as add_button(),
                # add_checkbox(), and set_status().

                # The icon unicode can be found at https://fontawesome.com/icons?d=gallery
                @view.dock.add_button(icon='f105')
                def next_waveforms_type(checked):
                    # The checked argument is only used with buttons `checkable=True`
                    view.next_waveforms_type()
