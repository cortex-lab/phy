# Example configuration for plugins shipped in a phy source checkout.
#
# Install the optional dependencies with:
#
#   uv sync --extra recluster
#
# Then copy this file to ~/.phy/phy_config.py. The checkout should remain the
# active phy installation so that the adjacent plugins directory is available.

from pathlib import Path

import phy

c = get_config()  # noqa: F821

repo = Path(phy.__file__).resolve().parents[1]
c.Plugins.dirs = [str(repo / 'plugins')]
c.TemplateGUI.plugins = ['ExampleReclusterPlugin']
