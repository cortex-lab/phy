# Troubleshooting

Here are a few tips if something goes wrong.


## GitHub issues

Look at existing [GitHub issues](https://github.com/cortex-lab/phy/issues) to see if someone else had the same problem. If not, feel free to open a new issue by giving a descriptive title and a comprehensive description of the problem, including screenshots if needed.


## Use the `--debug` option

When calling the phy GUI from the command-line interface, append the `--debug` option to the command to get more complete error messages, e.g. `phy template-gui params.py --debug`

Please copy and paste the full console output to the GitHub issue if requested.


## Sending your dataset to the main developer

In some cases, the [main phy developer](https://cyrille.rossant.net/) will ask you to send him your dataset in order to identify and fix the problem. You should compress the whole folder, **excluding**:

1. the `.phy` subdirectory, not needed, and
2. the raw data if it is too big.

Use a service like Dropbox, Google Drive, or Transfernow to send the archive to the developer.


## Common issues

### Graphic issue on Windows 10

Several users have reported display issues on Windows 10, especially on computers with a dual graphics chipset (e.g. integrated Intel GPU, and discrete NVIDIA GPU). phy works best on dedicated GPUs like NVIDIA cards. If you have this problem:

* Check that your screen is physically connected to your NVIDIA GPU and not the integrated one.
* Enable the discrete NVIDIA GPU (see [this user's advice](https://github.com/cortex-lab/phy/issues/922#issuecomment-561673363))

**Update (06/04/2020)**: a newer Intel Graphics driver version may fix the issue. See [this comment](https://github.com/cortex-lab/phy/issues/957#issuecomment-609498355).


### Issues with the GUI layout or the views

phy saves the GUI layout and view options (called **GUI state**) in two directories:

* **Global GUI state**: `~/.phy/TemplateGUI/state.json` for the Template GUI (common to all datasets)
* **Local GUI state**: `.phy/state.json` (just for a given dataset, inside your data directory)

If you want to reset the default GUI layout and view options, delete these two files, or run the GUI with the `--clear-state` option which will delete these files for you:

```
phy template-gui params.py --clear-state
```

You can also safely delete the `.phy` subdirectory which only contain the local GUI state and the data-specific cache, or use the `--clear-cache` option which will delete this directory for you:

```
phy template-gui params.py --clear-cache
```

More details:

* `.phy`: subdirectory inside the data directory. This subdirectory contains the cache that is used to make phy faster. You can always safely delete it: the cache will be automatically reconstructed the next time you launch the GUI. The only drawback is that performance will be a bit worse when you first select clusters in the GUI.

* `~/.phy`: (`~` is your home directory) this directory contains your custom plugins and user preferences for the GUI. If you delete it, you will lose the layout configuration of the GUI (which will be automatically reset the next time you open the GUI) and your user preferences (`~/.phy/phy_config.py`). More specifically, the GUI parameters are found in `~/.phy/TemplateGUI/state.json` for the Template GUI, and so on.


### Scaling discrepancy between templates and waveforms

There seems to be a scaling factor in the templates exported by KiloSort. Until this is fixed properly, a work-around is to add the following line to `params.py`:

``` python
template_scaling = 20.0  # or any other scaling factor
```


### Channel label inconsistency

The channel labels displayed in the views may be invalid. This may happen because the latest version of phy takes the `channel_map.npy` file into account when displaying the channel names in the views. If you want to disable this behavior, do the following:

1. Add `show_mapped_channels = False` in your `params.py` file.
2. Delete the `.phy` subdirectory within your data directory.
3. Launch phy again.


### Error "No module named 'PyQt5.QtWebEngineWidgets'"

Do:

```
pip install PyQtWebEngine
```

**Note**: make sure that PyQt5 is installed either with conda (default), or with pip, but not with both. Otherwise, conflicts may occur.


### Error "No module named PyQt5.sip"

If you receive the error: `No module named PyQt5.sip`, try to run the following commands in your conda environment (solution found by Claire Ward):

```
pip uninstall pyqt5 pyqt5-tools
pip install pyqt5 pyqt5-tools pyqt5.sip
```
