# Using the graphical interface

This page explains the interaction model shared by the Template GUI views. For a description of
each plot, see the [views reference](visualization.md). For a worked first session, start with the
[quickstart](quickstart.md).

## Main window

The main window contains the Cluster and Similarity tables, a menu and toolbar, a status bar, and
dockable graphical views. Use the **View** menu to add another view. Views can be resized, tabbed,
floated outside the main window, duplicated, or closed.

Each graphical view has controls for:

* opening its view-specific menu;
* taking a screenshot, saved by default under `~/.phy/screenshots/`;
* enabling or disabling automatic refresh after cluster selection changes;
* closing the view.

The arrangement and most view settings are restored the next time phy opens.

## Selecting clusters

The Cluster View contains the primary selection. The first selected cluster is blue, the second
red, and subsequent clusters use additional colors that are shared by the plots.

* Click a row to select it.
* Control-click or Command-click, depending on the platform's normal table behavior, to select
  additional rows.
* Shift-click to select a range.
* Control-right-click a Cluster View row to toggle it without discarding the rest of the
  selection.
* Type `:c 12 34` to select clusters by ID.

The Similarity View contains candidates relative to the primary Cluster View selection.
Control-right-clicking a Similarity View row promotes it into the primary selection while
preserving the current comparison. See [Similarity and the wizard](similarity.md) for the complete
workflow.

## Sorting and filtering

Click a Cluster View column header to sort the table. Enter a boolean expression in the filter box
and press Enter to restrict the visible clusters:

```text
group == 'good'
n_spikes > 10000
group != 'noise' && depth >= 1000
```

Press Escape to clear the filter. The `:s` and `:f` snippets provide keyboard-driven sorting and
filtering. Filtering and sorting also control the order and contents of global views such as the
Raster and Template views.

## Common plot interactions

Most graphical views share these controls:

* left-drag to pan;
* right-drag or use the wheel to zoom;
* double-click to reset pan and zoom;
* Control-wheel to change scaling where supported;
* Alt-wheel to change marker size in scatter-like views;
* Shift-wheel to change the color scheme in color-enabled views.

View-specific controls appear in the view menu and in the
[shortcut reference](shortcuts.md). Press `H` or use the Help menu to print the bindings active in
the current session.

## Automatic updates and large selections

Disable a view's automatic update when an expensive plot should remain fixed while you inspect
other clusters. Some views cap the number of selected clusters they display for responsiveness.
The Correlogram View, for example, displays the first 20 selected clusters by default. See
[Performance and spike sampling](performance.md) for the distinction between display limits and
spike-computation limits.

## GUI state

phy saves two kinds of state:

* global Template GUI state in `~/.phy/TemplateGUI/state.json`;
* dataset-specific state and cache in `<dataset>/.phy/`.

Dataset-specific values override global values. Use `--clear-state` to reset view layout and saved
settings, and `--clear-cache` to rebuild dataset computations:

```bash
phy template-gui params.py --clear-state
phy template-gui params.py --clear-cache
```

For more detail, see [Configuration and customization](configuration.md).
