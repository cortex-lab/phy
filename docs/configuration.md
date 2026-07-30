# Configuration and customization

Most users can work entirely through the GUI. Use configuration or a plugin when a setting must
apply repeatedly, a new metric or view is needed, or an existing action should behave
differently.

## Configuration locations

The user configuration directory is `~/.phy/`. Its main files and directories are:

```text
~/.phy/phy_config.py
~/.phy/plugins/
~/.phy/TemplateGUI/state.json
~/.phy/screenshots/
```

Each dataset may also contain `.phy/state.json` and cached computations under `.phy/`.

At startup, phy combines packaged defaults, global GUI state, and dataset-local state. Local state
has the highest precedence. Runtime changes are saved again when the GUI closes. This precedence
matters when a plugin changes a value that was previously saved: launch once with
`--clear-state`, and use `--clear-cache` as well when the setting changes a cached computation.

## Plugins

A plugin is a Python class derived from `phy.IPlugin`. Put plugin files in `~/.phy/plugins/` and
activate their class names in `~/.phy/phy_config.py`:

```python
c = get_config()
c.TemplateGUI.plugins = [
    'MyPlugin',
]
```

The filename is arbitrary, but plugin class names must be unique. See the
[plugin introduction](customization.md) and [plugin cookbook](plugins.md).

## Common customization tasks

* [Change keyboard shortcuts](keyboard_customization.md)
* [Change spike limits and performance settings](performance.md)
* [Understand or replace the similarity metric](similarity.md)
* [Add cluster metrics, columns, views, colors, and actions](plugins.md)
* [Use the in-process IPython console](visualization.md#ipython-view)

## Resetting safely

Use the narrowest reset that addresses the problem:

```bash
# Reset view layout and saved GUI options.
phy template-gui params.py --clear-state

# Rebuild cached dataset computations.
phy template-gui params.py --clear-cache

# Do both after changing a cached controller setting in a plugin.
phy template-gui params.py --clear-state --clear-cache
```

Clearing state does not delete curated `spike_clusters.npy` or cluster TSV files. Nevertheless,
back up custom files under `~/.phy/` before removing that directory manually.
