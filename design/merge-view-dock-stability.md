# Merge View dock and workspace stability plan

Status: implemented and automatically validated on the unreleased phy 2.2 branch;
manual dataset smoke testing and release acceptance remain

Companion documents:

- [Merge View workflow specification](merge-view-workflow.md)
- [Merge View architecture record](merge-view-architecture.md)
- [Merge Propositions specification](merge-propositions.md)

## 1. Goal

Changing merge propositions must feel like updating an existing workspace, not
closing and reopening a tool. Entering, cancelling, hiding, reopening, undoing,
or redoing Merge mode must retain the curator's dock placement and sizing choices
without disturbing unrelated widgets.

This plan changes presentation lifecycle only. The existing selection,
proposition, merge, history, save, and cancellation integrity contracts remain
authoritative.

## 2. Previous source of disruption

Before this plan was implemented, proposition transitions took a destructive GUI path:

1. activating another proposition calls `_cancel_merge_mode()`;
2. cancellation calls `_close_merge_view()`;
3. the Merge View dock has `Qt.WA_DeleteOnClose`, so the view and dock are
   discarded;
4. the next proposition constructs a new Merge View and dock; and
5. `_create_merge_view()` calls `QMainWindow.restoreState()` using a previously
   captured whole-window state.

This path is correct for curation state but unnecessarily reconstructs the Qt
dock layout. Whole-window restoration can also overwrite unrelated dock changes
made by the curator. Closing Merge View through its own close button follows a
different path and may not capture the same dock state before removal.

The resulting risks are:

- visible flicker while moving between propositions;
- changes to neighboring dock sizes and splitter proportions;
- loss of a floating dock's size or position;
- loss of docking area or tab relationships;
- repeated event connection and Qt resource lifecycle work; and
- different behavior for `V`, proposition navigation, automatic advancement,
  undo/redo, and the dock close button.

## 3. Target lifecycle

Merge View and its `QDockWidget` become persistent, dataset-scoped GUI objects.
They are created lazily on the first Merge-mode entry and destroyed only when
the Supervisor or GUI closes.

The lifecycle is:

```text
first Merge entry
    -> create view and dock once
    -> show and populate

manual Merge <-> proposition P1 <-> proposition P2
    -> reuse the same view and dock
    -> replace workspace contents atomically

cancel or dock close
    -> restore the Normal workspace
    -> hide the existing dock

later Merge entry
    -> show the same dock
    -> restore its prior local extent
    -> populate the new workspace

GUI shutdown
    -> disconnect callbacks
    -> close and release the persistent dock and view
```

The dock keeps one stable Qt object name throughout the dataset session. This
allows Qt to retain its docking area, floating state, tab group, and other local
layout metadata naturally.

## 4. Atomic workspace replacement

Moving from one proposition to another must not project an intermediate Normal
workspace into the GUI. The selection controller should expose an operation such
as `switch_merge_proposition(key, unit_ids)` that:

- requires an active Merge session;
- uses the active session's original Normal-entry snapshot;
- replaces proposition provenance, ordered staged IDs, and reference ID in one
  immutable selection transition;
- rebuilds the eligible Similarity rows once;
- preserves the Merge dock and its presentation state; and
- emits at most one settled selection projection.

The same operation should support replacing a manual Merge workspace with a
proposition. Its cancellation target remains the manual workspace's original
Normal-entry snapshot.

Automatic advancement after a successful proposition merge is slightly
different because the clustering has changed. It should construct the settled
post-merge Normal state, use that as the next proposition's entry snapshot, and
then project the next Merge workspace without hiding or recreating the dock.
Failed and manual merges retain their existing no-advancement behavior.

Reject-and-advance and shortcut navigation use the same in-place replacement
path. Selecting a nonactionable proposition still cancels to Normal mode and
hides Merge View because no review workspace remains active.

## 5. Persistent dock implementation

Replace the create/close pair with explicit lifecycle helpers:

- `_ensure_merge_view()` creates, configures, connects, and docks Merge View at
  most once;
- `_show_merge_view()` records the active state and reveals the existing dock;
- `_hide_merge_view()` hides it without removing it from `gui.views` or
  disconnecting its reusable interaction callbacks; and
- `_dispose_merge_view()` performs the current disconnection and release work
  during GUI shutdown only.

The persistent dock must not use `Qt.WA_DeleteOnClose`. Its close button becomes
a cancel-and-hide intent. The GUI's generic close handler must not remove the
hidden Merge View from `gui.views`.

Merge View contents remain ordinary projections of authoritative controller
state. Hiding the dock must not retain a second curation state inside the widget.
Showing it always refreshes its rows, reference, colors, status, drag policy, and
selection from the active selection state.

## 6. Geometry and size policy

Do not call whole-window `restoreState()` during Merge workflow transitions.
That operation affects every dock and can undo unrelated user layout changes.

The persistent dock should retain:

- docked versus floating state;
- floating position and size;
- dock area and tab relationship, maintained by the persistent Qt identity; and
- its most recent docked width or height.

Immediately before hiding, record the Merge dock's local extent. After showing a
docked view, use `QMainWindow.resizeDocks()` in the relevant orientation to
restore that extent. A floating dock retains and, if necessary, restores its own
`saveGeometry()` value.

Hiding a dock necessarily permits neighboring widgets to occupy the released
space. The requirement is therefore:

- proposition-to-proposition changes cause no dock movement at all; and
- hide/show may temporarily reflow visible widgets, but reopening restores the
  Merge dock's previous placement and proportions without permanently changing
  unrelated docks.

If strict zero reflow while inactive is ever required, Merge View must remain
visible in an inactive/empty state or occupy a reserved placeholder. That would
consume screen space and is not the default proposed here.

## 7. History and shutdown

Undo and redo continue to restore authoritative before/after selection and table
contexts. They should reveal, hide, or repopulate the persistent Merge View
without reconstructing its dock.

Keeping the view alive changes shutdown responsibilities. Supervisor close must
explicitly:

- disconnect Merge and Similarity drag/drop callbacks;
- disconnect dock close/cancel callbacks;
- remove any event-registry references owned by the persistent view;
- close the dock after workflow state has been settled for saving; and
- release Python references before interpreter shutdown.

This cleanup must preserve the existing regression protection for the
intermittent Qt shutdown crash.

## 8. Implementation sequence

1. Add characterization tests for dock identity and geometry across the current
   entry, cancellation, close-button, floating, and history paths.
2. Introduce `_ensure_merge_view()`, `_show_merge_view()`,
   `_hide_merge_view()`, and `_dispose_merge_view()` while retaining the current
   selection transitions.
3. Convert the dock close button and `V` cancellation to cancel-and-hide.
4. Add the atomic selection-controller transition for manual/proposition and
   proposition/proposition replacement.
5. Route click navigation, `Alt+Up`/`Alt+Down`, reject-and-advance, successful
   merge auto-advance, undo, and redo through the reusable view.
6. Remove transition-time whole-window `saveState()`/`restoreState()` calls and
   add local dock extent restoration.
7. Extend shutdown cleanup and leak/crash regressions for the persistent view.
8. Update the workflow specification, architecture record, user documentation,
   changelog, generated references when applicable, and PR description.

## 9. Required regression coverage

- `P1 -> P2` preserves `id(merge_view)` and `id(merge_view.dock)`.
- Manual Merge to proposition review preserves those identities.
- Shortcut navigation and reject-and-advance do not hide or recreate the dock.
- Successful auto-advance updates the existing view; failed and manual merges
  do not advance.
- All unrelated dock geometries remain unchanged across proposition switches.
- Cancel/hide/reopen restores the Merge dock area and docked extent.
- A floating Merge dock retains its exact position and size.
- The dock close button cancels and hides without removing the view.
- Repeated enter/cancel and proposition navigation do not multiply callbacks.
- Undo restores the exact pre-action Merge workspace in the same dock; redo
  restores the exact post-action workspace in that dock.
- Closing the GUI releases the persistent view, dock, Supervisor callbacks, and
  event-registry references without an intermittent Qt shutdown crash.

## 10. Acceptance criteria

The work is complete when:

- proposition navigation produces no visible dock/layout change;
- Merge View reopens where and at the size the curator left it;
- unrelated layout edits made while Merge View is hidden are preserved;
- every entry, cancellation, close, commit, rejection, reset, undo, and redo path
  uses one consistent dock lifecycle;
- curation and history regression suites remain green;
- repeated GUI lifecycle testing shows no retained Qt callbacks or shutdown
  crash; and
- `make lint`, `make format-check`, `make doc-check`, and `make test-full` pass.
