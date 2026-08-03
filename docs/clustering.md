# Manual clustering

phy lets you classify, merge, split clusters manually if the output of the automatic spike sorting algorithms are not satisfactory.

## Merging clusters

When multiple clusters seem to correspond to the same unit, select them and press `G` ("group") to merge them into a new cluster.

![image](https://user-images.githubusercontent.com/1942359/58953841-ded7b200-8797-11e9-9b2c-0b352c62999a.png)

All spikes belonging to either of the selected clusters will be assigned to that new cluster.

![image](https://user-images.githubusercontent.com/1942359/58953860-eac37400-8797-11e9-962d-2cf79ea55853.png)

### Staging candidates in Merge View

Press `V` with at least one Cluster View row selected to open **MERGE MODE**. phy stages the blue
reference first, followed by the other Cluster View selections and the current Similarity View
selection. Cluster View is dimmed and read-only while the workspace is active, but it remains
scrollable; its overlay reminds you that `V` returns to the normal workflow.

Every row in Merge View is part of the pending merge. Continue exploring Similarity View, where
sorting, filtering, `Control+Space`, and multi-selection remain available. The Merge View status
shows the number of staged clusters, selected similar clusters, and the total that `G` will merge.

Move candidates between the two views with `Control`-right-click or drag-and-drop. Drag within
Merge View to reorder candidates. The first blue reference cannot be moved or removed, and
the selection order shown by scientific views is always the Merge View order followed by selected
Similarity View rows in visible table order. Transfers, table sorting, and Merge reordering redraw
order-dependent views, while each cluster keeps the same color across tables and scientific views
for the entire Merge session. Press `Backspace` to clear only the Similarity View selection when the
merge should contain only the staged rows.

Press `G` to commit the staged clusters plus the selected Similarity candidates. Press `V` again,
use **Cancel Merge Mode**, or close Merge View to cancel and restore the exact state from before
entry. Undoing a committed Merge-mode merge restores the complete workspace as it appeared just
before `G`; Redo reapplies the merge and returns to the normal workflow.

### Reviewing merge propositions

For Template GUI datasets, phy can also review automatic merge propositions from a
dataset-local AIND/SpikeInterface format-version 2 `curation.json`. When the file
contains valid `merges`, the persistent **Merge Propositions** view labels source
entries `P1`, `P2`, and so on in their original file order, then shows a compact
ordered unit list: all IDs for four or fewer units, or the first two,
an ellipsis, the last, and the total count for larger propositions. A supplied
`new_unit_id` follows `⇒`; it is provenance only because phy allocates the result
through its ordinary merge model. There are no row action buttons. Hover a row
for its full IDs, key, status, reference, and any reason. Row colors identify
the active, accepted, accepted-modified, rejected, stale, and invalid lifecycle
states.

The `P` label is intended for navigation and discussion; filtering and status
changes do not renumber it. Review persistence continues to use the internal
stable key. Labels such as `P12.1` are reserved for future persisted propositions
derived from `P12`; this release does not generate derived proposition rows.

Click a pending row to stage it in Merge View immediately; this cancels and
replaces any manual or proposition workspace already open. Clicking a completed,
stale, or invalid row cancels any active workspace and only highlights that row.
You can still add, remove, and reorder candidates in a pending review. `Alt+Down`
and `Alt+Up` cancel the current workspace and open the next or previous pending
proposition in the current visible table order, wrapping at either end.
`Alt+Backspace` rejects the active proposition and advances; `Alt+Shift+Backspace`
resets the highlighted completed review and reopens it when reviewable.
These shortcuts do nothing while a text input has focus.

Press `G` to accept only after the ordinary merge succeeds; an edited set of
candidates is marked `accepted_modified`. A successful proposition merge opens
the next pending proposition using the visible order captured before the merge.
Manual and failed merges do not advance proposition review. Press `V`, use
Cancel, or close Merge View to leave the current proposition pending. Reject and
reset are undoable, and undo/redo restore the exact before/after proposition
workspaces. Overlapping propositions are allowed, but one whose source cluster
no longer exists after a merge or split becomes stale and is never automatically
remapped.

phy never overwrites producer-owned `curation.json`. It stores accepted and
rejected decisions in dataset-local `curation_review.json`, atomically after the
ordinary clustering files have been saved. Undo and redo restore both the
clustering result and the associated proposition review state.


## Splitting clusters

You can create a new cluster by drawing a polygon around a set of spikes in the feature view, the amplitude view, the template amplitude view, or the spike attribute views (**Control+click** to add points to the polygon).

![image](https://user-images.githubusercontent.com/1942359/58953705-8d2f2780-8797-11e9-8cca-e64567b9bb1b.png)

Then, press `k` ("kluster"). All spikes within the polygon are assigned to a new cluster.

![image](https://user-images.githubusercontent.com/1942359/58953725-9a4c1680-8797-11e9-9932-ad4ef57150d2.png)

Remaining clusters, i.e. spikes outside the polygon, are also assigned to new cluster ids. Remember that **cluster ids are unique** and are not reused when the clusters change.

Note: if not all spikes are displayed (there is a limit to the number of spikes displayed in each view), then all spikes are loaded before computing which spikes belong to the drawn polygon.

When exactly one cluster is selected in Normal mode, the Amplitude View also
supports a threshold split. Hold **Alt** and right-drag to place a horizontal
threshold, then press `K` to split the finite-amplitude spikes strictly below
it. The preview is sampled for responsiveness, but the committed split is
evaluated over every eligible spike in the cluster. Use
**Control+right-click** or **View > Clear amplitude split threshold** to clear
the threshold; an empty or whole-cluster threshold is not committed.


## Wizard

The **wizard** is a way to quickly get to pairs of clusters that might require merging.

You can move up and down in the **cluster view** with the `Up` and `Down` arrows. When using the wizard, the cluster selected in the cluster view is called the **best cluster**.

You can move up and down in the **similarity view** with the `Space` and `Shift-space` arrows. The cluster selected in the similarity view is called the **similar cluster**. The idea is to go through every "best cluster" in the cluster view, and review the "similar clusters" in the similarity view (sorted by decreasing similarity with the best cluster).

In Normal mode, ordinary single-row selection and wizard navigation both replace
the current similar cluster. With one blue reference, the replacement candidate
therefore remains red whether you click it or move with `Space`/`Shift+Space`.
Extending a selection with Control or Shift keeps existing cluster colors;
Control-deselecting and reselecting a row restores its reserved color. Backspace
clears those Normal-mode candidate reservations, so the next candidate starts
red again. Merge-mode colors remain fixed for the entire Merge session.

Press `Control+Space` to select the first 15 eligible clusters currently shown in the similarity view while preserving the cluster view selection. Repeat it to select the next batch. This uses the current similarity view sorting and filtering. To choose a different number, use **Select > Select N Similar**; the chosen number becomes the shortcut's new default and is remembered across sessions.

Wizard navigation skips clusters labeled `noise` or `mua` by default. To include them when moving through either table or when selecting N similar clusters, uncheck **Select > Skip Noise and MUA**. This preference is remembered across sessions. Direct selection with the mouse, a cluster ID, or a snippet can always select these clusters. Code that creates a `Supervisor` can choose the initial behavior with `skip_masked_clusters=False`; saved GUI state takes precedence when present.

On macOS, this shortcut uses the Control key, not Command. If `Control+Space` is assigned to switching input sources in macOS System Settings, disable or remap that system shortcut so that phy can receive it.

In Normal mode, scientific views follow the selected Cluster and Similarity rows in their visible
table order, with the blue Similarity reference first. Sorting or filtering either table updates
that presentation without recoloring existing table selections. Colors follow the selection
operation rather than incidental row order: replacement starts from the first candidate color,
while Control/Shift multi-selection preserves existing assignments. Selecting a new reference
starts a new color sequence. Use Merge mode when you need to collect or explicitly order candidates.

For each similar cluster, you can either:

* Press `space` to do nothing and go to the next similar cluster.
* Press `g` to merge the best and similar clusters, and go to the next similar cluster.
* Press one of the keyboard shortcuts to move either the similar cluster, the best cluster, or both clusters, to either the `good`, `mua`, or `noise` group (there are nine keyboard shortcuts for nine possibilities, see below). The best and/or similar clusters change automatically afterwards.
* Press `backspace` to unselect all similar clusters, and keep only best clusters (in the cluster view) selected.


## Moving clusters to different groups

Depending on the quality of the clusters, you can move them to the `good`, `mua`, or `noise` groups.

```
- move                                     - (:move)
- move_all_to_good                         ctrl+alt+g
- move_all_to_mua                          ctrl+alt+m
- move_all_to_noise                        ctrl+alt+n
- move_all_to_unsorted                     ctrl+alt+u
- move_best_to_good                        alt+g
- move_best_to_mua                         alt+m
- move_best_to_noise                       alt+n
- move_best_to_unsorted                    alt+u
- move_similar_to_good                     ctrl+g
- move_similar_to_mua                      ctrl+m
- move_similar_to_noise                    ctrl+n
- move_similar_to_unsorted                 ctrl+u
```


## Using cluster labels

phy supports custom cluster labels.

### Cluster label files

Cluster labels are saved in TSV (tab-separated values) files:

* Filename: `cluster_somename.tsv`
* Header: `cluster_id	somename` on the first line (there is a tab character between)
* Rows: `cluster_id	value` (for example, `47	good`)

A new column is automatically added for every cluster label TSV file found in the directory.

Cluster groups are saved in the same file format (`cluster_group.tsv`).

### Using labels in the GUI

You can also add cluster labels in the GUI. For example, to add a new label `neurontype` and assign the value `interneuron` to selected clusters:

* Select one or several clusters
* Press `:l neurontype interneuron` (this is the lowercase L snippet)
* Press `Enter`
* Save with **Control+S**

![image](https://user-images.githubusercontent.com/1942359/58955810-290f6200-879d-11e9-9fb0-06feb1268787.png)

A column is automatically added, and a `cluster_neurontype.tsv` file is automatically created with the following contents:

```
cluster_id	neurontype
299	interneuron
```


## Undo and redo

You can undo and redo clustering actions (merge, split, move, label) with the **Control+Z** and **Control+Y** keyboard shortcuts.
