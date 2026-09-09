# Merge workflow feedback: implementation handoff

Status: approved and implemented on 2026-09-06; automated validation is recorded
below and real-dataset acceptance remains.

## Goal and scope

Support the curator feedback about reference removal, manual candidate collection,
table search, consistent mouse gestures, and column rearrangement. Consolidate
workflow operations and regression-test complete sequences so future changes do
not break selection, colors, merge contents, cancellation, or history.

The user requested an assessment and this durable plan. No implementation was
made, and this note does not authorize publishing, messaging reviewers, or merging
PRs. GitHub state was not checked during the assessment.

Read `AGENTS.md`, the issue audit when working on issues, and the existing
workflow, architecture, proposition, and selection-order/color documents before
implementation. This plan proposes changes to some existing contracts; update
those documents alongside implementation rather than silently contradicting them.
The dated integration handoff contains historical release gates, not verified
current GitHub state.

## Verified current behavior

| Request | Evidence in the current checkout |
| --- | --- |
| Transfer the blue Merge reference into Similarity | `_selection.py:CurationSelectionController.remove_from_merge` rejects it. `deselect_from_merge` already promotes the next staged cluster, but removes the old reference from effective selection instead of transferring it. MergeView also prevents dragging the reference. |
| Rearrange table columns | `gui/widgets.py:Table` uses native Qt headers, without movable sections or persisted column order. MergeView disables header sorting, independently of column layout. |
| Type an ID to select it | The editor accepts click focus and has editing tests. It applies expression filters on Enter. A bare nonzero integer is a truthy constant, not an ID lookup. MergeView hides the editor; ClusterView is disabled in Merge mode. |
| Ctrl+left-click multiselection | Already implemented and tested in ClusterView and SimilarityView. Merge rows are staged members, not an independent multiselection. |
| Transfer candidates in Normal mode | Table Ctrl+right-click transfers only in Merge mode. `test_supervisor_control_right_click_transfers_only_in_merge_mode` explicitly asserts the Normal-mode no-op. Ordinary right-click is consumed without transferring. |
| Transfer from correlograms | `views/correlogram.py:on_mouse_release` emits a deselection request on stationary Ctrl+right-click. `apps/base.py:create_correlogram_view` resolves the target: diagonal ID, or Similarity member of a mixed-role pair. It is neither row-based nor a transfer. |

Useful implementation locations:

- `phy/cluster/_selection.py`: immutable state, transitions, reference and palette invariants.
- `phy/utils/selection.py`: selection intents and mutations.
- `phy/cluster/supervisor.py`: MergeView, transfer callbacks, projection through
  `_apply_selection_change`, Merge lifecycle and history integration.
- `phy/gui/widgets.py`: table events, filtering, sorting and Qt headers.
- `phy/cluster/views/correlogram.py` and `phy/apps/base.py`: plot hit testing and
  application-level interpretation.
- Tests: `phy/cluster/tests/test_selection.py`, `test_supervisor.py`,
  `test_merge_lifecycle.py`, `phy/gui/tests/test_widgets.py`,
  `phy/cluster/views/tests/test_correlogram.py`, `phy/apps/tests/test_base.py`.

## Approved behavior

These decisions were approved during implementation.

1. Keep Ctrl+left-click for Normal-mode multi-(de)selection. Ordinary right-click
   transfers; Ctrl+right-click is not a compatibility alias. Stationary plot
   clicks transfer and drags do not.
2. Define primary role as Cluster selection in Normal mode and staged Merge
   membership in Merge mode. Transfer the clicked cluster between that role and
   Similarity. A plot click targets the row cluster, including off-diagonal cells.
   Unselected table rows transfer directly without a preliminary left-click.
3. Allow reference transfer when another primary cluster remains. Promote the
   next remaining primary cluster deterministically, retain the old reference in
   Similarity, and recompute similarity against the new reference. Reject transfer
   of the last primary cluster with clear feedback; cancellation
   remains the explicit way to leave Merge mode.
4. Preserve the blue-reference invariant. Existing promotion swaps the promoted
   cluster's palette slot with slot zero. Reuse a documented policy for transfers
   and deselection, preserving other bindings and the original entry snapshot.
5. Support the manual sequence: select A in ClusterView; explore Similarity;
   transfer B and C into Cluster selection; Backspace clears remaining Similarity;
   V stages A/B/C. The feedback's step 5 says SimilarityView, but context indicates
   ClusterView. V already supports manual entry.
6. Define exact-ID lookup separately from expression filtering. A
   bare integer plus Enter selects and reveals that ID; expressions retain filter
   behavior. Missing IDs are non-mutating, a filtered-out ID clears the expression
   so it can be centered, and the lookup follows the enabled table's replacement
   selection semantics. Do not stage candidates or change
   Merge membership through generic search. No live-as-you-type selection is assumed.
7. Add independent column order for Cluster, Similarity, and Merge tables. Persist
   by column name, tolerate added/removed plugin columns, and keep logical sorting
   separate from visual column order. The ID column is movable but not hideable.

`G` merges the complete effective selection: the unique union of Merge and
Similarity selections, including selected rows hidden by filtering.

## Implementation sequence

1. Re-read the current code and specifications; record the gesture/operation table
   for both modes, including reference and last-member cases. Add focused failing
   regressions for the requested behavior; intentionally revise tests that assert
   superseded restrictions.
2. Extend the existing controller with explicit role-transfer transitions and a
   shared reference-promotion helper. Keep select, deselect, transfer and reorder
   distinct. Preserve validation, immutable snapshots and `SelectionChange`.
   Avoid a new global state framework or broad Supervisor rewrite.
3. Route table, drag/drop and correlogram transfer requests through the same
   Supervisor/controller operations. Move workflow decisions out of the plot
   application callback. Project each completed transition once; preserve event
   compatibility where possible and test any changed event contract.
4. Implement and test ID lookup/focus semantics as a contained change.
5. Implement column movement and persistence separately; it need not block core
   workflow fixes. Reordering columns must not reorder staged clusters.
6. Update workflow/architecture/proposition contracts as needed, shortcuts, user
   documentation and the unreleased changelog. Perform full validation and a
   real-dataset smoke test before declaring these workflows ready.

## Regression requirements

Controller tests should verify that transfers of active IDs preserve effective
membership, assign the destination role without duplicate ownership, preserve
unrelated selections, and use deterministic order and color behavior. Do not require
Normal-mode roles to be disjoint without auditing existing overlap semantics. Test
reference promotion, absent IDs, repeated requests, last-member rejection, and invalid
operations leaving state unchanged.

GUI tests must use actual mouse/key events as well as direct controller calls:

- Ctrl+left toggles, right-click transfers one intended row, and plot drags do not
  transfer. Check platform modifier mapping and both halves of the CCG matrix.
- Equivalent table and plot operations yield equivalent workflow state.
- Filtering/sorting preserve hidden selections and palette reservations; delayed
  or stale table events cannot overwrite a newer transition.
- Lookup handles focus, Enter/Escape, valid/missing/hidden IDs and expressions.
- Column movement survives reopen, leaves sorting correct, tolerates plugin
  column changes, and preserves selection and staged order.

Sequence tests must cover both proposition and manual entry, with several
transfers including the original reference, Backspace, further exploration, and:

- cancellation restoring the complete entry selection, colors and workflow context;
- commit using the explicitly agreed operands, returning to the merged result for
  quality assignment, and correct undo/redo of the workspace;
- failed merge preserving the workspace and assignments;
- edited propositions retaining the correct identity and review/history semantics;
- save/reopen preserving actual curation and not leaking transient staging;
- repeated entry/exit without callback or GUI-resource accumulation.

Use table-driven sequences or a small reference model for transition coverage.
Retain meaningful integration tests; controller tests alone cannot catch Qt focus,
gesture handling, debounce, or cross-view projection regressions. Keep interactive
work proportional to selected candidates where possible, not all dataset spikes.

## Validation baseline and completion

Assessment command (after successful `uv sync --dev`):

```bash
uv run pytest -q phy/cluster/tests/test_selection.py phy/gui/tests/test_widgets.py phy/cluster/tests/test_supervisor.py phy/apps/tests/test_base.py -k 'selection or merge or filter or correlogram_deselect'
```

Result: **96 passed, 131 deselected**, 26.78 seconds. Shutdown emitted pending
ipykernel task messages despite exit code zero. This was a focused automated run,
not full-suite or real-dataset acceptance. No source changes were made. Existing
untracked `site/` was left untouched.

For implementation, run narrow tests while iterating, then `make lint`,
`make format-check`, `make test-full`, and `make doc-check`. Run `uv build` if
packaging/dependencies/entry points/package data change. Document actual results
and limitations. Manually exercise both entry workflows on a dataset copy,
including reference transfer, remaining Similarity candidates at commit,
cancel/undo/redo, quality assignment, and save/reopen. Passing old tests is not
evidence that the newly requested behavior works.

Implementation validation: the focused controller, widget, Supervisor,
Correlogram, and application suite passed **236 tests**. `make lint` and
`make format-check` passed. Documentation generation, link checking, and strict
MkDocs building passed; the final clean-tree comparison reports the intended
uncommitted documentation changes. The core/GUI half of `make test-full` passed
**483 tests**. Its application half passed **152 of 153 tests**; the unrelated
`test_template_controller_without_templates_uses_stored_waveform_channels`
fails both in the full run and isolation because phylib reports
`n_samples_waveforms == 0` instead of the fixture's expected 20. No code on that
waveform-loading path was changed here. A real-dataset smoke test was not possible
without a supplied dataset copy.
