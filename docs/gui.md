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

### View settings

Open a plot's view-specific menu and choose **View settings** to edit related
parameters together:

* Waveform, Amplitude, and Correlogram views expose independent per-cluster
  spike budgets and an optional shared total budget.
* The Amplitude View also exposes its gray background-spike budget.
* The Correlogram View also exposes bin size, window size, and refractory
  period.
* Firing Rate and ISI views expose bin size and the displayed range. These
  views use every spike in the displayed clusters and therefore have no
  spike-sampling budget.

Enabled budget checkboxes map directly to integer controller settings.
Disabling one maps that setting to `None`. Changes apply and replot
immediately. The existing snippets and mouse-wheel controls remain available
for quick single-setting changes.

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

Press `C` to stage the current selections in Merge View. In this temporary mode, Cluster View is
disabled, every Merge View row is included in the pending merge, and Similarity View remains
available for exploring additional candidates. Control-right-click or drag rows between Merge and
Similarity views, or drag inside Merge View to reorder candidates. Press `G` to commit or `C` to
cancel. See [Staging candidates in Merge View](clustering.md#staging-candidates-in-merge-view).

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

Spike-budget controller settings are global and carry across datasets.
Correlogram bin/window/refractory settings and Firing Rate/ISI bin/range
settings are dataset-specific, because appropriate display scales can depend
on the recording.

For more detail, see [Configuration and customization](configuration.md).
