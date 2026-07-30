# Quickstart: your first ten minutes

This walkthrough uses the Template GUI, the recommended interface for
template-based sorters such as KiloSort. It shows how to inspect a dataset,
compare clusters, make a small curation change, and save it safely.

Manual curation changes the sorter's output. Work on a copy until your lab has a
tested archival workflow.

## 1. Open a dataset

### Try the small example

The phy test-data repository contains a compact Template GUI dataset:

```bash
git clone https://github.com/kwikteam/phy-data.git
cd phy-data/template
phy template-describe params.py
phy template-gui params.py
```

The example is intended for learning and testing the interface, not for deriving
a scientific curation policy.

### Open your own sorting output

First make a backup outside the output directory. On Linux or macOS:

```bash
cp -a /path/to/output /path/to/output-before-phy
```

On Windows PowerShell:

```powershell
Copy-Item -Recurse C:\path\to\output C:\path\to\output-before-phy
```

Then describe and open the dataset:

```bash
phy template-describe /path/to/output/params.py
phy template-gui /path/to/output/params.py
```

If `template-describe` fails, fix the input before curating. The
[dataset guide](dataset.md) explains the required arrays and `params.py`.

## 2. Learn the selection model

The GUI is built from dockable **views**. You can move, resize, tab, close, and
reopen them from the View menu. The most important distinction at first is:

- **Cluster View** lists the current clusters and their metadata. Its selected
  cluster is the unit you are evaluating.
- **Similarity View** ranks candidates relative to the Cluster View selection.
  Selecting a row there adds a candidate for comparison.
- **Waveform, Correlogram, Feature, Amplitude, Firing Rate, Probe, and Trace
  views** update from the current selection when their auto-update option is on.

Click one row in the Cluster View. Then click a highly ranked row in the
Similarity View. The other views should now compare the two clusters using
different colors.

Press `H` at any time to show the effective keyboard shortcuts. Menus also show
their shortcuts, and hovering over an action shows its command name in the status
bar.

## 3. Inspect one cluster

Start with one Cluster View row and ask a few concrete questions:

- Does the waveform look spike-like and localized on neighboring channels?
- Does its shape remain reasonably consistent across individual spikes?
- Does its autocorrelogram have a plausible refractory-period dip around zero?
- Is its amplitude stable through the recording, or is there drift or an abrupt
  discontinuity?
- Do the raw traces, when available, agree with the extracted waveforms?

No single plot proves that a cluster is a well-isolated neuron. Low spike counts
also make correlograms noisy. Treat the views as complementary evidence and use a
curation policy agreed for your experiment.

Useful first-session keys include:

| Action | Default shortcut |
| --- | --- |
| Show all shortcuts | `H` |
| Select the next similarity candidate | `Space` |
| Return to only the Cluster View selection | `Backspace` |
| Merge selected clusters | `G` |
| Split a feature selection | `K` |
| Undo / redo | `Ctrl+Z` / `Ctrl+Shift+Z` |
| Save | `Ctrl+S` |

On macOS, menu labels may use platform-native key names. The help window is the
authoritative list for the running build.

## 4. Compare a possible merge

With a cluster selected in the Cluster View, press `Space` or choose a candidate
from the Similarity View. Similarity is a ranking aid, not a merge decision.
Compare at least:

- waveform shape and spatial footprint;
- cross-correlogram versus both autocorrelograms;
- feature-space overlap, when PC features are available;
- amplitude and firing behavior over time;
- the raw trace around representative spikes, when available.

If the evidence strongly supports one unit split by the sorter, select the
clusters and press `G` to merge. phy gives the result a new cluster ID. Press
`Ctrl+Z` immediately if the result is not what you intended.

Splitting requires selecting spikes in a view that supports lasso or polygon
selection, commonly the Feature View, and pressing `K`. It is worth learning
merge, undo, and save on the example dataset before attempting a scientific
split.

## 5. Assign a group

The standard cluster groups are `good`, `mua`, `noise`, and `unsorted`. With a
Cluster View row selected:

| Group | Default shortcut |
| --- | --- |
| Good | `Alt+G` |
| MUA | `Alt+M` |
| Noise | `Alt+N` |
| Unsorted | `Alt+U` |

These shortcuts apply to the primary Cluster View selection. The Edit menu also
contains actions for the Similarity View selection and all selected clusters.
Check the menu wording before using a broader action.

Labels are experimental judgments, not properties inferred by phy. Define what
`good`, `mua`, and `noise` mean for your analysis before curating a full
recording.

## 6. Save and understand the output

Press `Ctrl+S` or choose **File → Save**. The important files written in the
dataset directory are:

- `spike_clusters.npy`: the current cluster assignment for every spike;
- `cluster_group.tsv`: the group assigned to each current cluster;
- other `cluster_<field>.tsv` files for additional labels;
- `cluster_info.tsv`: a convenient snapshot of the columns currently exported
  from the Cluster View.

The original `spike_templates.npy` is not changed by merges or splits.
`cluster_info.tsv` is a derived summary; use `spike_clusters.npy` and the
`cluster_<field>.tsv` files as the primary curation results.

phy does **not** automatically make a backup before overwriting these files.
Keep the pre-curation copy made above, and consider dated snapshots or version
control for subsequent milestones. Undo and redo are session history, not a
replacement for saved backups.

Closing with pending changes should prompt you to save or discard them. Do not
rely on that prompt as your only safeguard; save deliberately after a small,
verified batch of work.

## Cache, state, and logs

phy also creates files that are not curation results:

- `.phy/` inside the dataset contains its local GUI state and computation cache;
- `~/.phy/TemplateGUI/state.json` contains GUI state shared across Template GUI
  datasets;
- `phy.log` in the dataset directory records diagnostic messages.

Deleting `.phy/` does not delete `spike_clusters.npy` or cluster metadata, but
the next launch may be slower while caches are rebuilt. To reset a troublesome
cache or layout from the command line:

```bash
phy template-gui params.py --clear-cache
phy template-gui params.py --clear-state
```

`--clear-cache` removes the dataset `.phy` cache. `--clear-state` resets both
dataset-local and global Template GUI state. The latter affects layouts and
persisted view settings for other datasets too, so use it intentionally.

For an error report, launch with:

```bash
phy template-gui params.py --debug
```

Include the console traceback, relevant part of `phy.log`, operating system,
Python and phy versions, installation method, and whether plugins are enabled.

## Where to go next

- [Preparing a dataset](dataset.md) explains every common input category.
- [Visualization](visualization.md) covers views, navigation, and snippets.
- [Clustering](clustering.md) covers merge, split, labels, undo, and save.
- [Customization](customization.md) introduces configuration and plugins.
- [Troubleshooting](troubleshooting.md) covers common startup, state, and graphics
  problems.
