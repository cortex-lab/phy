# Troubleshooting

Here are a few tips if something goes wrong.

## GitHub issues

Look at existing [GitHub issues](https://github.com/cortex-lab/phy/issues) to see if someone else had the same problem. If not, feel free to open a new issue by giving a descriptive title and a comprehensive description of the problem, including screenshots if needed.

If you are reporting a problem with `phy 2.1.0`, please also include:

* operating system
* Python version
* installation method
* local or remote session details
* whether plugins are in use
* a minimal error message or reproduction if possible

## Use the `--debug` option

When calling the phy GUI from the command-line interface, append the `--debug` option to the command to get more complete error messages, e.g. `phy template-gui params.py --debug`

Please copy and paste the full console output to the GitHub issue if requested.

## Standalone graphics diagnostic

For frozen or blank plots, the repository provides a standalone
[graphics diagnostic script](https://github.com/cortex-lab/phy/blob/master/scripts/diagnose_graphics.py).
It works with an existing installation; no new phy release is required. It has
been exercised against phy 2.1.0 and a development checkout.

You can give a coding agent this short prompt:

> Follow the instructions in [the graphics diagnostic script](https://github.com/cortex-lab/phy/blob/master/scripts/diagnose_graphics.py).

For reproducible support requests, a maintainer can substitute a commit permalink
for the branch link. The script contains the agent instructions, diagnostic code,
and interpretation guidance. The agent locates your existing phy environment,
runs synthetic graphics tests, investigates findings with selectable follow-up
probes, and returns a ZIP to send to the maintainer. Use an agent with terminal
access to the affected machine and graphical session. A cloud-only session tests
its own environment. Tool approvals depend on the agent's configuration; the
diagnostic itself requires no user feedback.

To run it yourself, download the raw Python file, activate the environment you
normally use for phy, and run `python diagnose_graphics.py` from your affected
desktop session. Small test windows will open. The script compares available
Linux backend settings in separate processes and writes `phy-graphics-*.zip`
in the current directory. Use `--baseline-only` to skip the comparisons.

The ZIP contains a summary, structured findings and results, logs, synthetic PNGs,
and the exact diagnostic script. A scene probe uses the installed ClusterView,
Supervisor, WaveformView, and CorrelogramView with deterministic synthetic data
and an automated Qt mouse click. Images include capture type and expected content
so an agent can inspect them where its tools support images. The script captures
only its synthetic surfaces/widgets, not the whole desktop or any recordings.
Home/current-directory paths are redacted on a best-effort basis. Inspect the
bundle before sharing if local paths are sensitive. No packages are installed
and user phy settings are not changed; scene state is stored in a temporary directory.
These checks do not establish correctness with a real dataset or prove desktop
presentation. An SSH run characterizes that session and may differ from the
affected desktop.

Agents can query `python diagnose_graphics.py --list-probes` for JSON capabilities.
`--workspace DIR` preserves attempts, logs, images, and findings across calls;
`--probe`, `--configuration`, `--scenario`, and `--repeat` select bounded follow-ups.
For example, `--probe docked --configuration baseline --scenario hide-show` tests
one docking operation. `--budget` bounds each invocation and crashes/hangs retain
available fault-handler output. The agent writes its interpretation and evidence
references to `AGENT_NOTES.md` in the workspace, then uses
`--workspace DIR --bundle --output report.zip` to package all attempts without
rerunning them. Script measurements and agent interpretations remain distinct.

Frame checks record the original redraw request, a bounded first/last frame
history, a Qt timer heartbeat, window/container exposure, context lifecycle and
ownership, viewport/framebuffer geometry, and intended versus actual pan/zoom.
Incorrect early frames can recover naturally; a later incorrect frame still
invalidates the observation. A separately recorded recovery redraw never erases
the original outcome. Hidden/unexposed surfaces or inconsistent sampling return
an inconclusive result instead of an unexplained colour failure.

Input-event traces contain event kinds and wheel deltas, without key values or
text. `--input-mode block` isolates external input to the synthetic canvases.
`--capture-mode metrics` skips PNG capture/encoding to investigate timing effects;
it retains primitive frame measurements but reduces scene coverage. These are
control experiments, and passing them does not establish normal interaction.

Maintainers can validate controlled faults in a real desktop session with
`PHY_DIAGNOSTIC_GL_TESTS=1 uv run pytest scripts/tests/test_diagnose_graphics.py`.
Those opt-in tests deliberately inject a wrong colour, wheel input, or a hidden
canvas and verify the expected failure/inconclusive classifications and PNG hashes.
Linux CI runs the controlled faults through xcb/software OpenGL as well. The
remaining standalone tests run without opening native GL surfaces.

Interrupted collections can resume their workspace without deleting lock files;
unrecorded outcomes remain marked as interrupted or unknown. Missing or damaged
images are omitted with reasons while the other logs and results are preserved.
Reports retain historical failures and distinguish consistently passing alternative
configurations from a claim that an intermittent failure has been fixed.

### Frozen or blank plots on Chromebook Linux / Wayland

A Chromebook Linux user running phy 2.1.0 reported that selecting another cluster
updated the tables while plots remained frozen and zoom stopped responding.
Launching with the matched X11/GLX settings below restored selection updates and
zoom for that user. This is a tested workaround for that installation; the exact
cause of the original Wayland freeze has not been established.

Close phy, then run this from the dataset directory in the same environment:

```bash
QT_QPA_PLATFORM=xcb QT_XCB_GL_INTEGRATION=xcb_glx PYOPENGL_PLATFORM=glx phy template-gui params.py --debug
```

These settings apply to this launch only. They do not install packages or reset
dataset or GUI state. Keep the normal launch command on systems where it works.
If X11 is unavailable or this launch fails, retain the log and use the standalone
diagnostic rather than changing packages or clearing settings.

Setting only `QT_QPA_PLATFORM=xcb` may leave PyOpenGL using Wayland's EGL binding
while Qt uses GLX. Repeated `Attempt to retrieve context when no valid context`
errors accompanied blank plots in the affected user's X11 test. The same error
was reproduced with deliberately mismatched bindings on another Linux system.
It is evidence of a context lookup failure, not proof of its cause on every system.

Phy preserves existing backend settings and explicit user overrides. It does not
switch every Chromebook or Wayland installation automatically: Qt may fall back
to EGL even when GLX was requested, and the original freeze has not been
reproduced sufficiently to choose a safe automatic activation rule.

The diagnostic's `--configuration xcb-glx` experiment sets all three variables
and works with existing releases. The `xcb` experiment remains a Qt-only control.
Selection screenshots now wait for selection-triggered frames without requesting
extra redraws; a passing synthetic run still does not verify physical desktop
presentation, real datasets, or user plugins.

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

### Error "No module named PyQt5.sip"

If you receive the error: `No module named PyQt5.sip`, try to run the following commands in your conda environment (solution found by Claire Ward):

```
pip uninstall pyqt5 pyqt5-tools
pip install pyqt5 pyqt5-tools pyqt5.sip
```
