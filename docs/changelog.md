# Changelog

This file records user-visible changes to phy. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) where practical.

## [Unreleased] — 2.1.1.dev0

Changes below are available from the latest source checkout but have not yet
been included in a stable release. The current entries cover all user-visible
changes committed on 23 July 2026; test-only commits are represented by the
behavior they verify rather than listed separately.

### Added

- Select the first eligible clusters in the Similarity View with
  `Control+Space`. The default is 15 clusters; **Select > Select N Similar**
  changes the number and remembers it across sessions.
- Skip clusters labeled `noise` or `mua` during wizard navigation and batch
  similarity selection. **Select > Skip Noise and MUA** controls the behavior
  and remembers the preference across sessions.
- Use `Control`-right-click to add or remove a Cluster View row from the
  current selection, or to promote a Similarity View row while preserving the
  existing selections and similarity reference.
- Right-click a cross-correlogram to promote its similar cluster into the
  Cluster View selection. Native mouse and trackpad secondary clicks are
  supported.
- Configure the total number of gray background points in the Amplitude View
  with `n_spikes_amplitudes_background` (10,000 by default).
- Waveform, Amplitude, and Correlogram views use fixed total spike budgets
  across multi-cluster selections, while retaining their per-cluster ceilings.

### Changed

- Views with a cluster-display limit now show the first eligible clusters in
  the current selection instead of retaining stale view contents. The
  Correlogram View displays at most 20 selected clusters by default.
- Row actions preserve the intended Cluster View and Similarity View
  selections.
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
