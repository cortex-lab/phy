
# Example phy configuration file.
#
# Copy this file to ~/.phy/phy_config.py to enable the plugins that ship with
# this repository (in particular ExampleReclusterPlugin, plugins/recluster.py).
#
#   Windows : copy phy_config.example.py %USERPROFILE%\.phy\phy_config.py
#   Linux / macOS : cp phy_config.example.py ~/.phy/phy_config.py
#
# The plugin path is derived from the installed phy package rather than
# hardcoded, so this file works unchanged on any machine that clones the repo,
# as long as phy is installed editable from it:
#
#   pip install -e .[recluster]
#
# (the [recluster] extra pulls in scikit-learn and isosplit, which the plugin
# needs). You can also put your own plugins in ~/.phy/plugins/.

import pathlib

import phy
from phy import IPlugin

# Plugin example:
#
# class MyPlugin(IPlugin):
#     def attach_to_cli(self, cli):
#         # you can create phy subcommands here with click
#         pass

c = get_config()

# Point phy at the plugins folder that ships with this repo. phy.__file__ resolves
# to <repo>/phy/__init__.py, so parents[1] is the repo root and <repo>/plugins is
# the folder next to this file -- no hardcoded, user-specific path.
_repo = pathlib.Path(phy.__file__).resolve().parents[1]
c.Plugins.dirs = [str(_repo / 'plugins')]

# Discovery alone is not enough: phy only *attaches* plugins listed here by name.
# ExampleReclusterPlugin adds the recluster actions (alt+k / shift+alt+k / ctrl+alt+k)
# to the Template GUI.
c.TemplateGUI.plugins = ['ExampleReclusterPlugin']
