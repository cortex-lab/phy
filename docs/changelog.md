# Changelog

This file records user-visible changes to phy. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) where practical.

## [Unreleased] — 2.2.0.dev0

Changes below are available from the latest source checkout but have not yet
been included in a stable release. The current entries cover all user-visible
changes committed since 23 July 2026; test-only commits are represented by the
behavior they verify rather than listed separately.

### Documentation

- Documented uv-first installation of exact previous phy releases in separate
  environments, including how to intentionally replace a tool installation.

### Added

- `Control`-right-clicking a diagonal autocorrelogram in the Correlogram View
  removes that cluster from the active selection. On a cross-correlogram spanning
  the primary and Similarity selections, it removes the Similarity cluster. This
  also works during Merge mode and proposition review; removing the reference
  promotes the next staged cluster to reference.
- Split the lower-amplitude portion of one selected cluster directly from the
  Amplitude View: use `Alt`-right-drag to preview a threshold, then press `K`
  to commit an exact all-spike split. Individual waveform traces receive the
  same transient preview; **Control+right-click** or **View > Clear amplitude
  split threshold** clears it.
- Display elapsed recording time in seconds, minutes, or hours in Amplitude
  and Firing Rate views. Choose the shared preference from **View > Recording
  time unit** and see open views update immediately. Axis labels use thousands
  separators and at most two decimal places.
- Stage and order manual merge candidates in the new **Merge View**. Press `V`
  to enter or cancel Merge mode, transfer candidates with
  `Control`-right-click or drag-and-drop, and press `G` to merge every staged
  cluster plus the current Similarity View selection. Merge View opens below
  Cluster View and keeps one stable in-session dock identity, position, and
  size; proposition navigation updates that dock without moving neighboring
  views. The dimmed Cluster View remains scrollable. Scientific views follow Merge View order and then
  selected Similarity rows in visible table order. Cancellation restores the entry state, and
  undo restores the full pre-merge workspace.
- Review AIND/SpikeInterface format-version 2 merge propositions from
  dataset-local `curation.json` in a persistent **Merge Propositions** view.
  Its compact rows have no action buttons and carry stable source-order display
  labels (`P1`, `P2`, ...). The selection background marks the active review;
  foreground colors show completed/problem states while blue stays reserved for
  the merge reference cluster. Click a pending row to stage it in Merge View,
  while tooltips retain the full IDs, status, key, and reason.
  `Alt+Down`/`Alt+Up` navigate pending rows, `Alt+Backspace` rejects and advances,
  and `Alt+Shift+Backspace` resets a
  highlighted completed review. `G` accepts the ordinary merge, marks edited
  acceptance `accepted_modified`, and opens the next pending proposition in the
  pre-merge visible order. Undo/redo restore exact proposition workspaces; stale
  overlapping proposals are never remapped, and decisions are atomically saved
  in `curation_review.json` without overwriting `curation.json`.

- Select the first eligible clusters in the Similarity View with
  `Control+Space`; repeat the shortcut to select successive batches. The
  default is 15 clusters; **Select > Select N Similar** changes the number
  and remembers it across sessions.
- Skip clusters labeled `noise` or `mua` during wizard navigation and batch
  similarity selection. **Select > Skip Noise and MUA** controls the behavior
  and remembers the preference across sessions.
- Configure the total number of gray background points in the Amplitude View
  with `n_spikes_amplitudes_background` (10,000 by default).
- Waveform, Amplitude, and Correlogram views support optional fixed total spike
  budgets across multi-cluster selections. They use fixed per-cluster budgets
  by default so sampling accuracy does not decrease as selections grow.
- Configure per-cluster and optional shared spike budgets from **View
  settings** in the Waveform, Amplitude, and Correlogram view menus.
  Correlogram, Firing Rate, and ISI settings dialogs also expose their bin,
  window, range, and related display parameters.
- Hidden or inactive-tab Waveform, Amplitude, and Correlogram views defer
  selection plotting until they become visible, retaining only the latest
  selection.

### Fixed

- Keep the Merge Propositions table layout and scroll state stable while moving
  between pending propositions instead of rebuilding the full queue.
- Release GUI, Supervisor, table, dock, and curation event callbacks when a
  dataset window closes, preventing retained Qt widgets and intermittent
  process crashes during shutdown.
- Closing the Merge View now restores staged clusters to their original
  Cluster and Similarity View rows, selections, and table positions. Reopening
  reveals the same dock at its prior docked extent or floating geometry.
- Show the active sort column and direction in Cluster and Similarity View
  headers.
- Dragging Merge View rows now shows the cluster ID preview, insertion boundary,
  edge autoscroll, and a row-wide hover cue without changing the order until
  the drop completes.
- Show an unsaved-changes marker in the window title, enable the Save action
  only when curation changes are pending, and confirm successful saves in the
  status bar.
- Rename the Help shortcut reference action to **Show shortcuts and commands**
  and show Enter/Escape guidance when the `:` command prompt is active. The
  shortcut now opens an in-GUI searchable reference, including plugin actions.
- Pressing `:` repeatedly no longer leaves the command prompt visible after
  Escape closes it.
- Keep the disabled Cluster View overlay fixed while scrolling in Merge mode,
  increase its dimming, and make native table rows initiate drag-and-drop.
- Display Firing Rate View values in spikes per second instead of normalized
  probability density, with the configured bin count matching the rendered bins.
- Start the GUI with released phylib versions that do not yet expose the
  disjoint-spike selection optimization hint.
- Keep dataset-local view settings isolated from global GUI state. In
  particular, a Firing Rate time range saved or leaked from another recording
  no longer clips spikes in a fresh dataset.
- Cluster and Similarity View filters only take keyboard focus after an
  explicit click, including when a table is first shown or refreshed. Their
  native double-click text selection remains editable. Enter, Escape, and
  outside clicks release filter focus so global shortcuts resume.
- Display metadata columns containing multiple values in the Cluster and
  Similarity Views instead of leaving their cells blank.

### Changed

- Put content-specific actions first in every view menu, followed by a
  consistent Auto-update, Screenshot, and Close utility footer.

- Move Trace View scale shortcuts from `Alt+Up`/`Alt+Down` to
  `Control+Alt+Up`/`Control+Alt+Down`, leaving `Alt+Up`/`Alt+Down` available for
  Merge Propositions navigation.

- Group cluster traversal commands under **Select > Navigation**.

- Group available views under **View > Add view** and keep global view options
  separate from view creation.

- The first, blue Cluster View selection is now the explicit Similarity
  reference. In Normal mode, scientific views follow the selected Cluster and
  Similarity rows in visible table order; sorting or filtering either table
  updates that presentation without recoloring existing selections. Color
  assignment now follows explicit selection intent: ordinary clicks and
  `Space`/`Shift+Space` replace the Similarity candidate and reuse the first
  candidate color, so a lone candidate remains red; Control/Shift
  multi-selection preserves existing colors and reserves a toggled-off row's
  color for reselection; Backspace releases Normal-mode candidate reservations.
  Choosing a new reference starts a new color session. In Merge mode, explicit
  Merge View order takes precedence and all existing colors remain fixed for
  the entire session. Normal-mode cross-role mouse transfers and
  cross-correlogram promotion have been removed in favor of the Merge workspace.
- Undo and redo restore the complete selection context around merge, split,
  and metadata actions; redo also preserves selection-only exploration made
  after the original action.
- Merge and assignment operations update the small cluster-ID collection
  incrementally instead of rescanning every spike.
- Merges gather their spikes from the maintained per-cluster arrays while
  preserving the previous globally sorted spike order. Code that directly
  mutates the exposed `spike_clusters` array must explicitly rebuild the
  clustering indexes before invoking an operation.
- Native tables establish their fixed row height from the first populated
  payload instead of measuring every row after each update.
- Native tables retain column widths established by their first populated
  payload instead of rescanning all cells after later updates. Exceptionally
  long replacement values may therefore be clipped.
- Merge-related Cluster and Similarity View mutations batch expensive column
  and row fitting until the complete table update is applied.
- Views with a cluster-display limit now show the first eligible clusters in
  the current selection instead of retaining stale view contents. The
  Correlogram View displays at most 20 selected clusters by default.
- Row actions preserve the intended Cluster View and Similarity View
  selections.
- `Control`-left-click consistently adds or removes a row from the selection
  in its current Cluster or Similarity View.
- On macOS, the `Control+Space` batch-selection shortcut uses the physical
  Control key rather than Command.
- Cluster and Similarity View selections use a shorter debounce interval,
  reducing the delay before dependent views update.
- Spike selection uses the disjoint-cluster fast path, reducing selection
  overhead on large datasets.
- Template and feature amplitude points use stable, evenly spaced samples.
  Background clusters share one fixed display budget, so the number of gray
  points no longer grows with the number of eligible clusters.
- Waveform plotting reuses shared coordinate bounds and time axes, and draws
  each channel axis once, reducing geometry and temporary allocations for
  multi-cluster selections.

### Fixed

- Keep keyboard focus in the cluster filter while its results update.
- Preserve Similarity View column widths when the table refreshes.
- Use the live clustering after merges and splits when uncapped spike times
  are requested.
- Handle native table secondary clicks and trackpad secondary clicks
  consistently.
- Clear stale Similarity View selections when wizard navigation changes the
  primary cluster.
- Hide the feature amplitude option when a dataset has no feature arrays.

### Documentation

- Added beginner guides for installation, dataset preparation, first launch,
  curation, saving, and troubleshooting.
- Documented the default similarity metric, keyboard-shortcut customization,
  view computation limits, waveform pre-extraction and its subset limitation,
  performance tuning, CLI options, configuration, outputs, and advanced
  workflows.
- Added source-install instructions for fresh environments using the latest
  `phy` and `phylib` commits on Linux, macOS, and Windows.
- Added a task-oriented documentation navigation, synchronized API, shortcut,
  and plugin references, and strict documentation checks in continuous
  integration.

## [2.1.0] — 2026-07-17

This maintenance release restored reliable installation and operation on
current systems while leaving dataset and file formats unchanged.

### Added

- Added a project Code of Conduct.

### Changed

- Modernized packaging, dependencies, and continuous integration for Python
  3.10 and newer.
- Replaced the legacy web-based Cluster View with a native Qt implementation.
- Improved GUI startup, rendering, and resource cleanup on Linux, macOS, and
  Windows.

### Fixed

- Preserved Cluster View sort order when cluster metadata changes
  ([#1375](https://github.com/cortex-lab/phy/issues/1375)).
- Displayed NumPy scalar values correctly in native Qt Cluster View columns
  ([#1377](https://github.com/cortex-lab/phy/issues/1377)).

See the [2.1.0 release notes](release.md) for installation and
compatibility details.

## [2.0a1] — 2019-06-17

This alpha began the phy 2 series. It introduced the current plugin-oriented
curation interface, new cluster visualization views, improved splitting and
filtering workflows, multiple raw-data-file and shank support, and the
`cluster_info.tsv` export. It also moved shared non-graphical utilities to
phylib and updated the GUI to PyQt5.

## [1.0.0] — 2016-03-24

Historical stable release of the original phy interface. Development leading
to this tag expanded manual clustering, visualization, plugin support, and
template-model workflows.

## [0.2.2] — 2015-09-10

Historical maintenance release with fixes and incremental improvements to
clustering, data loading, and GUI behavior.

## [0.2.1] — 2015-07-11

Historical patch release focused on automatic-clustering, trace, and
cross-platform fixes.

## [0.2.0] — 2015-07-10

Historical feature release that expanded the Kwik-based workflow, plotting,
spike detection, automatic clustering, and public API documentation.

## [0.1.0] — 2015-05-26

First tagged phy release.

Complete commit-level history is available in the
[GitHub tags and releases](https://github.com/cortex-lab/phy/tags). The
repository did not maintain complete structured release notes for every older
tag, so the summaries above are intentionally concise.

[Unreleased]: https://github.com/cortex-lab/phy/compare/v2.1.0...HEAD
[2.1.0]: https://github.com/cortex-lab/phy/releases/tag/v2.1.0
[2.0a1]: https://github.com/cortex-lab/phy/compare/v1.0.0...v2.0a1
[1.0.0]: https://github.com/cortex-lab/phy/compare/v0.2.2...v1.0.0
[0.2.2]: https://github.com/cortex-lab/phy/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/cortex-lab/phy/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/cortex-lab/phy/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/cortex-lab/phy/releases/tag/v0.1.0
