"""Show how to change the default text font size."""

from phy import IPlugin
from phy.plot.visuals import TextVisual


class ExampleFontSizePlugin(IPlugin):
    def attach_to_controller(self, controller):
        # Smaller font size than the default (6).
        TextVisual.default_font_size = 4.
