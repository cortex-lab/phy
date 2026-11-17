# Exact parent restoration and inspection

The curation workflow offers **Restore parents in Merge Mode** for a live unit
with exact recorded immediate ancestry. **Inspect parents (read-only)** remains
a separate operation. Restoration changes real assignments; inspection creates
isolated providers and changes no curation state.

## Restoration and workspace history

`Clustering.restore_partition()` validates a complete, disjoint, nonempty
partition of the live target and allocates fresh cluster IDs. It uses cached
target rows and native assignment/history events, avoiding a recording-wide
search for affected spikes. Historical IDs are provenance, never resurrected
live identities. Merge rows show a **Parent** column beside the current ID.

`Supervisor.restore_parents()` appends one new global-history action, with a
rollback snapshot limited to affected assignment rows plus metadata/history and
workflow state. It does not remove an earlier history entry or rewind later
work. New memberships clear metadata rather than inherit the merged unit's
quality. Native undo-state metadata snapshots also preserve exact parent values
when a later remerge is undone.

The post-restoration Merge session contains actual live units. Its entry snapshot
is a Normal selection of the fresh parents; the history before-snapshot describes
the old merged target. Consequently V/cancel leaves the restored units intact,
Undo restores the merged unit without reversing earlier unrelated edits, and
Redo restores the parent workspace. Existing Merge operations remain available;
Normal-mode split/quality/label tools remain accessed through V.

**Exclude current row** acts on the native Merge table's current row and removes
that ID from the effective merge selection. It is intentionally distinct from
right-click transfer: transferred Similarity units remain selected and are still
included by G. Tests exercise actual row/button clicks and unchanged transfer
semantics.

## Ordinary ancestry and durability

`ParentLineage` retains exact immutable per-parent spike memberships captured at
merge time. `_phy_merge_parents.npz` is atomic, non-pickle and bound to saved
assignments, spike times and templates. Save retains ancestry reachable from live
units and origin aliases, including retired intermediate composite parents.
Fresh restored children can therefore refer back to their historical merge and
restore another level after save/reopen. Unreachable ancestry can be discarded.

Optional color records preserve pre-merge palette slots; unavailable historical
colors use a deterministic fallback. The origin map and restoration retirement
projection survive reopen. SLAY acceptance decisions retain their historical
result IDs; an accepted result subsequently restored is shown as such instead
of appearing to be a current accepted merge or becoming pending automatically.

Legacy merges without recorded exact membership are unavailable. Split
descendants are not silently treated as their former merged ancestor. Every
partition is checked against the current unit before mutation; original
sorting templates are never used to guess a missing partition.

## Bounded journal and full-source replay

Bounded restoration appends a source- and prefix-bound version-3
`restore_parents` event. It references an earlier merge in the same preparation,
its currently active equivalent target, ordered fresh child IDs, and explicit
metadata clearing. Referenced membership is reconstructed immediately before
that merge, including any earlier source-amplitude predicate expressions.
Nested restored composite children retain equivalent-membership origin links.

The native change, semantic event, metadata, review projection, selection and
history participate in one rollback transaction. A source parent with no retained
evidence cannot currently be represented as an editable native unit, so GUI
restoration rejects it before mutation rather than silently omit it. Legacy
merge-capable packages can use the command; explicit capability denial is
honored. Older bounded consumers reject the new event version/capability.

Replay uses explicit event dispatch instead of sending restoration journals down
the merge-only mapping shortcut. Existing source/union/split expression trees
represent restored memberships and continue to drive live summaries, indexes
and source-unit resolution. Historical predicate projection uses adaptive
sub-batches with a bounded mask allowance, not parent-count times the complete
recording. The independent output contract reconstructs pre-merge masks in
bounded chunks and verifies metadata clearing separately from the runtime.

Save/reopen in the same package preserves ancestry. Replay exports restored live
units, metadata, reviews and retirement provenance for further preparation.
A fresh preparation starts from its new source assignments and does not retain
recoverable membership anchors for pre-preparation historical merges; those
merges remain unavailable. No claim of arbitrary cross-round historical
restoration or arbitrary sampled lasso replay is made.

## Read-only inspection

Inspection uses an embedded dark Qt workspace containing a QTableWidget parent
catalog and separate native correlogram, waveform, amplitude and feature views.
Providers operate only on the exact requested parent memberships, respecting
per-view sample budgets. Bounded correlograms describe retained contiguous
windows and have no full-recording rate baseline. Missing stored/raw waveforms
can use explicitly labelled, capped template traces without changing membership.

The workspace snapshots and restores dock layout, visibility, actions and focus;
Back/Esc and a new selection dispose it without pumping queued curation events.
Main-window close restores the original layout before geometry saving. Inspection
supports at most 16 immediate parents; that display limit never truncates actual
restoration. Curation actions and inherited editing shortcuts are blocked during
inspection, and its view handlers do not publish global editing/transfer events.

## Validation

Focused regressions cover later unrelated edits, exact fresh-ID assignments,
metadata clearing and undo preservation, nested ancestry, invalid partitions,
rollback failures, real exclusion gestures, save/reopen, immediate review status,
bounded earlier predicate parents, zero-evidence rejection, replay dispatch,
summary expression equivalence, scratch limits and unsupported old consumers.

Rendered copied-data walkthroughs and independent full-source replay evidence
are recorded in `carl-feedback-captures/README.md`. These are engineering checks
of the supplied converted fixture, not validation of its acquisition clock,
exposure assumptions or biological merge quality.
