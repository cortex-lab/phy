# Troubleshooting

Here are a few tips if something goes wrong.


## GitHub issues

Look at existing [GitHub issues](https://github.com/cortex-lab/phy/issues) to see if someone else had the same problem. If not, feel free to open a new issue by giving a descriptive title and a comprehensive description of the problem, including screenshots if needed.


## Sending your dataset to the main developer

In some cases, the [main phy developer](https://cyrille.rossant.net/) will ask you to send him your dataset in order to identify and fix the problem. You should compress the whole folder, **excluding**:

1. the `.phy` subdirectory, not needed, and
2. the raw data if it is too big.

If you don't include the raw data in the compressed archive, please let the developer know the exact filename and file size (in bytes) of the raw data file(s).

Use a service like Dropbox, Google Drive, or Transfernow to send the archive to the developer.


## Common issues

### Graphic issue on Windows 10

Several users have reported display issues on Windows 10, especially on computers with a dual graphics chipset (e.g. integrated Intel GPU, and discrete NVIDIA GPU). phy works best on NVIDIA GPUs. If you have this problem:

* Check that your screen is connected to your NVIDIA GPU and not the integrated one.
* Enable the discrete NVIDIA GPU (see [this user's advice](https://github.com/cortex-lab/phy/issues/922#issuecomment-561673363))

### Deleting the `.phy` and `~/.phy` subdirectories.

The cache directories might sometimes cause problems. Deleting them may help. You can also use the `--clear-cache` and `--clear-state` options to the `phy` command.

phy uses two user directories to store user parameters and cache:

* `.phy`: subdirectory inside the data directory. This subdirectory contains the cache that is used to make phy faster. You can always safely delete it: the cache will be automatically reconstructed the next time you launch the GUI. The only drawback is that performance will be a bit worse when you first select clusters in the GUI.

* `~/.phy`: (`~` is your home directory) this directory contains your custom plugins and user preferences for the GUI. If you delete it, you will lose the layout configuration of the GUI (which will be automatically reset the next time you open the GUI) and your user preferences. More specifically, the GUI parameters are found in `~/.phy/TemplateGUI/state.json` for the Template GUI, and so on.
