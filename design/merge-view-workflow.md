# Merge View workflow specification

Status: implemented and automatically validated on the unreleased phy 2.2 branch;
manual dataset smoke testing and release acceptance remain

Companion document: [Merge View architecture record](merge-view-architecture.md)

This document fixes the user-facing behavior of the manual Merge View workflow.
Merge Propositions extend this contract in
[their own specification](merge-propositions.md).

## Purpose

Merge View provides a temporary workspace tied to one reference cluster. It
allows a curator to collect and order plausible merge candidates while continuing
to inspect candidates from Similarity View. Entering, editing, or cancelling the
workspace does not change clustering data. Only the existing merge action, `G`,
commits a curation change.

## Modes and effective selection

The GUI has two mutually exclusive modes:

- **Normal mode:** the effective selection is the union of the Cluster View and
  Similarity View selections. Scientific views show the blue reference first,
  followed by the other selected Cluster View rows and selected Similarity View
  rows in their visible table order.
- **Merge mode:** the effective selection is the union of every cluster in Merge
  View and the Similarity View selection. Cluster View is disabled.

The effective selection is what graphical and scientific views display. It is
also what `G` merges. Transferring a cluster between Similarity View and Merge
View does not change effective membership. It publishes a render update only
when the transfer also changes presentation order.

Merge mode is active exactly while Merge View contains its blue reference
cluster.

## Entering Merge mode

`V` enters Merge mode. At least one cluster must be selected in Cluster View. If
there is no Cluster View selection, the action does nothing and reports why.

Before changing the UI, phy snapshots the complete state needed to restore the
normal workflow on cancellation. At minimum, this includes:

- Cluster View selection and its order;
- Similarity View selection and its order;
- the Similarity reference;
- presentation order and cluster colors; and
- any filter, sort, scroll, or navigation state changed by entering the mode.

All selected clusters are transferred into Merge View in this order:

1. the blue reference cluster;
2. the remaining Cluster View selections, in their existing order; and
3. the Similarity View selections, in their existing order.

After the transfer:

- Cluster View and Similarity View have no selected rows;
- Cluster View remains visible and scrollable but is clearly read-only: it
  cannot be selected, navigated, filtered, sorted, or used as a drag source or
  target;
- Similarity View remains enabled and remains calculated relative to the blue
  reference; and
- the effective selection and graphical displays initially remain unchanged.

Staged clusters are not offered again as active Similarity candidates while they
remain in Merge View.

## The blue reference

Merge View is tied to one blue reference cluster:

- it is always the first row;
- it remains the reference used to calculate Similarity;
- it cannot be reordered;
- it cannot be removed individually; and
- its blue color is stable throughout the workflow.

When several clusters were selected in Cluster View, the cluster that was the
Similarity reference on entry becomes the blue reference. The other clusters
remain ordinary Merge candidates and retain their colors.

## Merge View membership and ordering

Merge View has no selected-versus-unselected row state. Every cluster present in
the view is included in the pending merge by definition. Removing a row means
removing that cluster from Merge View.

- Ctrl+right-click on a non-reference Merge row transfers that cluster to
  Similarity View and selects it there.
- Ctrl+right-click on a Similarity row transfers that cluster to Merge View.
- Drag-and-drop between Similarity View and Merge View provides the equivalent
  transfer operation.
- Dragging within Merge View reorders candidates, except for the fixed blue
  reference.
- New candidates are appended after the candidates already in Merge View unless
  they are dropped at an explicit insertion point.
- Duplicate membership is not possible; adding an existing member is a no-op.

Ctrl+click retains its existing multi-row selection meaning in Similarity View.
Ctrl+right-click transfers only the clicked row. Dragging a selected Similarity
row transfers all selected rows; dragging an unselected row transfers only that
row.

In Merge mode, the presentation order delivered to scientific views is always
the Merge View row order followed by selected Similarity View rows in visible
table order.
Adding, removing, or reordering rows may redraw order-dependent scientific
views, but a cluster's color slot remains fixed across workflow tables and
scientific views for the entire Merge session.

## Exploring Similarity

Similarity View remains the live candidate area during Merge mode:

- Ctrl+Space keeps its existing behavior and selects the next configured number
  of eligible Similarity candidates.
- Ctrl+click manages its multi-row selection.
- Backspace clears only the Similarity View selection; it does not alter Merge
  View.
- Filtering, sorting, and Similarity navigation remain available.

The GUI must continuously and prominently show the pending merge count, for
example:

```text
MERGE MODE — 4 staged + 2 selected similar = 6 clusters
```

This makes it clear that `G` includes selected Similarity candidates. To merge
only the contents of Merge View, the curator presses Backspace before `G`.

## Committing a merge

There is one merge shortcut in both modes: `G`.

In Merge mode, `G` merges the unique union of:

- every cluster in Merge View; and
- every selected cluster in Similarity View.

If the union contains fewer than two clusters, no curation action occurs and the
GUI reports that another candidate is required.

After a successful merge:

- Merge mode ends;
- Merge View is cleared and hidden while its dataset-scoped dock is retained;
- Cluster View is re-enabled;
- the new merged cluster becomes the blue Cluster View selection; and
- Similarity View is recomputed for the new cluster using the normal post-merge
  workflow.

If the merge fails, the complete Merge-mode state remains unchanged.

## Cancelling Merge mode

All of the following cancel Merge mode:

- pressing `V` while Merge mode is active;
- activating a prominent **Cancel Merge Mode** control; or
- closing Merge View.

Cancellation performs no clustering action. It restores the exact snapshot from
immediately before Merge mode was entered, regardless of additions, removals, or
reordering performed in Merge mode. In other words:

```text
state A -> enter Merge mode -> edit workspace -> cancel -> state A
```

Closing Merge View must visibly communicate that it cancels the mode. Merge mode
must also be unmistakable while active: Merge View is labelled **MERGE MODE**,
Cluster View is dimmed or overlaid with an explanation, and the status area shows
the pending merge count.

Merge View and its dock are created lazily once per dataset session. Cancelling,
closing, undoing, or re-entering Merge mode hides, reveals, and repopulates that
same dock, preserving its placement and size without restoring the whole-window
layout.

## Undo and redo

Before committing a Merge-mode merge, phy records the workspace state immediately
before `G`, including:

- ordered Merge View cluster IDs;
- the blue reference;
- cluster-to-color assignments;
- selected Similarity cluster IDs; and
- the relevant Similarity state.

Undoing that merge restores both the original clusters and the complete Merge
workspace as it existed immediately before `G`. Redoing it reapplies the merge
and exits Merge mode again. This special restoration applies only to merges
initiated from Merge mode; ordinary merge undo behavior remains unchanged.

Workspace transfers and reordering are temporary UI operations and do not create
entries in the clustering undo stack.

## State transitions

| State | Action | Result |
| --- | --- | --- |
| Normal | `V` | Snapshot state, transfer all selections, enter Merge mode |
| Merge | Ctrl+right-click Similarity | Transfer clicked candidate to Merge |
| Merge | Ctrl+right-click removable Merge row | Transfer candidate to Similarity |
| Merge | Ctrl+Space | Select the next Similarity candidates |
| Merge | Backspace | Clear only the Similarity selection |
| Merge | `G` | Merge Merge contents plus selected Similarity candidates |
| Merge | `V`, Cancel, or close Merge View | Restore the entry snapshot exactly |
| After Merge-mode merge | Undo | Restore clusters and pre-commit Merge workspace |
| Restored after undo | Redo | Reapply merge and return to normal mode |

## Extension

The [Merge Propositions specification](merge-propositions.md) defines how
external propositions enter this workspace without changing the manual workflow
contract.
