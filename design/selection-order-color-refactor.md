# Selection ordering and color-state refactor

Status: implementation specification

## 1. Motivation

Cluster selection currently has three partially independent representations:

- role membership and `presentation_order` in `CurationSelectionState`;
- a mutable `_selection_color_order` in `Supervisor`; and
- selected rows and color-index mappings projected into tables and scientific
  views.

This split causes subtle inconsistencies. In Normal mode, deselecting one
Similarity row compacts the mutable color order and recolors later clusters.
Color order is also absent from cancellation and history snapshots, so restoring
a Merge workspace can restore the correct cluster IDs without necessarily
restoring their original colors.

The implementation should correct the state model rather than add more
mode-specific conditions to `Supervisor._update_selection_colors()`.

## 2. Scope

This refactor covers:

- Normal- and Merge-mode role membership;
- the explicit Similarity reference;
- scientific-view presentation order;
- stable selected-cluster color slots;
- table sorting and filtering;
- Merge entry, transfer, reorder, cancellation, commit, undo, and redo; and
- projection into workflow tables and built-in scientific views.

It does not introduce a global reactive store, change spike-level clustering
algorithms, or require a new public selection-event payload.

## 3. Authoritative state

`CurationSelectionState` is the sole authoritative selection and rendering
state. It owns:

```python
mode
cluster_ids
similar_ids
reference_id
presentation_order
color_order
merge
```

`presentation_order` contains the active cluster IDs in the exact order sent to
scientific views. `color_order` is a reference-scoped registry: its tuple
position is the selected-cluster palette slot. It may retain inactive cluster
IDs so deselection and reselection do not change colors.

### 3.1 State invariants

1. All role, presentation, and color sequences contain unique cluster IDs.
2. `set(presentation_order) == set(effective_ids)`.
3. `set(effective_ids) <= set(color_order)`.
4. If a reference exists, it belongs to the active primary role and occupies
   index zero in both `presentation_order` and `color_order`.
5. Normal mode has no Merge session and its reference belongs to `cluster_ids`.
6. Merge mode has no Cluster role selection, and the reference is the first
   Merge member.
7. In Merge mode, `presentation_order` begins with the exact Merge View order;
   its tail contains exactly the selected Similarity IDs.
8. A non-empty Similarity selection requires a reference.
9. Selection transitions perform work proportional to cluster IDs, never to
   spike count.

## 4. Color lifecycle

Color slots are stable for the lifetime of one reference:

- Selecting a previously unseen cluster appends it to `color_order`.
- Deselecting a cluster removes it from active presentation but retains its
  color slot.
- Reselecting a cluster reuses its existing slot.
- Sorting, filtering, Merge transfers, and Merge reordering never change color
  slots.
- Editing the Cluster selection while retaining the same reference preserves
  existing slots and appends new clusters.
- Changing the reference starts a new color session. The new reference becomes
  slot zero, and active clusters receive fresh slots in presentation order.
- Entering Merge mode preserves the Normal color registry.
- Cancelling Merge restores the complete entry registry.
- A committed merge selects a new reference and therefore starts a new registry.
- Undo and redo restore the exact registry stored in their selection snapshots.

Color slots are not reused before the reference changes. The registry is bounded
by the number of clusters encountered for one reference and remains independent
of the number of spikes.

## 5. Presentation lifecycle

Presentation and color order are independent.

- Normal presentation is the explicit reference first, followed by selected
  Cluster rows and selected Similarity rows in their table order, without
  duplicates.
- Merge presentation is Merge View order followed by selected Similarity rows
  in Similarity View table order.
- A filter does not deselect hidden rows. Visible selected rows come first in
  visible order; filtered-out selections remain at the tail in their previous
  relative order.
- Sorting and filtering may change presentation and redraw order-dependent
  scientific views, but never recolor clusters.

Because table order is external UI state, `presentation_order` remains stored in
the immutable state and is restored by cancellation and history.

## 6. Controller transitions

The selection controller should expose explicit transitions for distinct user
intents:

```python
set_normal_selection(...)
set_similarity_selection(...)
set_presentation_order(...)
enter_merge_mode(...)
cancel_merge_mode()
add_to_merge(...)
remove_from_merge(...)
reorder_merge(...)
restore(...)
```

`set_presentation_order()` validates ordering without changing membership or
colors. Supervisor code must not normalize presentation by invoking a second
membership transition.

`SelectionChange` should distinguish:

```python
roles_changed
presentation_changed
colors_changed
reference_changed
mode_changed
render_changed  # presentation_changed or colors_changed
```

### 6.1 Snapshot simplification

`NormalWorkflowSnapshot` should store the complete immutable Normal selection
state plus opaque table workflow context, rather than duplicate individual
selection fields. If changing the dataclass layout in the first implementation
step would make the migration unnecessarily risky, adding `color_order` to the
existing snapshot is an acceptable intermediate commit; the duplicated fields
must still be removed before completing the refactor.

## 7. Supervisor responsibilities

The Supervisor translates table intents, invokes one controller transition, and
projects the resulting state. It does not independently own color state.

Required changes:

1. Remove `Supervisor._selection_color_order`.
2. Make `Supervisor.selection_color_order` delegate to
   `selection.state.color_order`.
3. Remove the `reset` policy from `_update_selection_colors()`; projection uses
   the state's color order verbatim.
4. Replace `_normalize_presentation_order()` with a pure table-order
   calculation followed by `set_presentation_order()`.
5. Canonicalize selection intent before projection so one user operation
   produces one authoritative transition.
6. Handle both `table_sort` and `table_filter` through the same presentation
   reorder path.
7. Keep one `_apply_selection_change()` path responsible for selected-row
   projection, Similarity refresh when required, table colors, Merge View,
   task logging, and scientific-view publication.
8. Publish when `render_changed` is true, while retaining the existing public
   `emit('select', supervisor, cluster_ids)` positional payload.

Programmatic table projection must remain non-emitting. Revision checks continue
to reject delayed events from an obsolete table state or workflow mode.

## 8. View projection

Workflow tables receive `state.color_order` through
`set_selected_index_order()`. Built-in scientific views obtain an immutable copy
of the same mapping for each selection render and use it only for palette lookup;
layout continues to follow `presentation_order`.

Standalone views without a Supervisor may fall back to positional colors.
Attached built-in views must not silently fall back when an authoritative color
mapping exists but omits an active cluster; that condition indicates a violated
state invariant and should be covered by tests.

## 9. Required regression coverage

### 9.1 Normal mode

- Select A, then C, then B: presentation follows A/B/C table order while C
  retains its color.
- Select a Ctrl+Space batch, deselect a middle row, and verify all remaining
  colors are unchanged.
- Reselect the removed row and verify its original color returns.
- Select a new row after a deselection and verify it receives a new slot.
- Sort and filter both role tables without recoloring.
- Modify Cluster selection without changing the reference and retain colors.
- Change the reference and verify the new reference is blue and slots reset.

### 9.2 Merge mode

- Enter Merge with the complete Normal presentation and color registry.
- Select and deselect Similarity candidates without recoloring.
- Transfer candidates in both directions without recoloring.
- Reorder Merge rows without recoloring.
- Sort and filter Similarity without recoloring.
- Cancel by shortcut, button, and view close and restore exact entry state.

### 9.3 History and actions

- Commit a Merge, undo it, and restore the exact pre-commit roles,
  presentation, colors, and blue reference.
- Redo and restore exact post-commit state.
- Preserve selection-only exploration, including its color registry, across
  undo/redo.
- Restore exact color state around ordinary merge, split, and metadata actions.

### 9.4 Cross-view consistency

- Workflow-table color indices match authoritative color slots.
- Representative per-cluster views such as Waveform and Correlogram match.
- Representative global/vectorized views such as Template and Raster match.
- Trace and scatter views match.
- Each intent emits at most one public selection update.

## 10. Implementation sequence

Each commit must leave focused tests passing.

1. `test: characterize reference-scoped selection colors`
   - Add failing Normal deselection/reselection and Merge undo color tests.
2. `refactor: store color order in selection state`
   - Add state invariants, transition logic, snapshot/history restoration, and
     remove Supervisor-owned color policy.
3. `refactor: make presentation reordering explicit`
   - Add `set_presentation_order()`, remove membership-changing normalization,
     and cover sorting/filtering.
4. `refactor: centralize selection projection`
   - Consolidate Supervisor intent handling and projection/publication.
5. `refactor: simplify selection snapshots and color projection`
   - Remove duplicated snapshot fields and tighten built-in view mapping.
6. `test: cover ordering and colors across lifecycle`
   - Complete the regression matrix and event-count assertions.
7. `docs: document reference-scoped color behavior`
   - Update changelog, clustering documentation, API output, and architecture.

## 11. Validation

During implementation, run the narrowest relevant tests. Before handoff run:

```bash
uv run pytest phy/cluster/tests/test_selection.py
uv run pytest phy/cluster/tests/test_supervisor.py
uv run pytest phy/cluster/views/tests
uv run pytest phy/apps/tests/test_base.py
make lint
make format-check
make doc-check
make test-full
```

The working tree may contain concurrent changes. Every commit must inspect the
index explicitly and include only files belonging to its implementation phase.
