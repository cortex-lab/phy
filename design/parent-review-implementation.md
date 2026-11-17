# Exact merge-parent inspection

Implementation branch: `carl/parents`; base `534972f4`.

`BaseController.parent_inspection` owns lineage and a separate read-only comparison
window. Cluster View's row context menu offers **Show parents**. The View menu
also offers it for a single selected unit. The comparison has an explicit
**Return to merged unit** toolbar action and closes on the next selection,
curation mutation, main GUI closure, or controller closure. Opening from another
row selects the merged unit first; display-only parent IDs are never published
as a Supervisor selection.

The comparison creates its own correlogram, waveform, amplitude, and feature
providers, over exact parent spike arrays. Original providers, selectors,
clustering, labels, history and save actions are never replaced. Views are added
without `ManualClusteringView.attach`, so they register no global selection,
split, or channel callbacks and cannot invoke curation. A legend identifies each
parent as `child ← parent`; a shared parent-to-color mapping is used across
views, including when a parent has no waveforms or feature rows. Full global
callback-registry equality is tested across repeated opening and closure.

## Lineage

Ordinary `Clustering._do_merge` retains the pre-merge `spikes_per_cluster` array
references on its UpdateInfo. Clustering replaces those arrays rather than
mutating them. The tracker therefore adds no duplicate spike buffers during
merge. Nested merges and merges following splits retain immediate parent
membership exactly. Undo does not overwrite a recorded partition; redo captures
the same pre-merge grouping. Retired records are retained in memory for undo and
only live merged units are written on successful Save.

Ordinary `_phy_merge_parents.npz` is an atomic, non-pickle sidecar bound to the
saved assignment vector, spike times and original template vector. A stale or
invalid sidecar is refused rather than reinterpreted. It is independent of the
software-only curation provenance log. Merges predating this feature have no
recoverable exact ordinary history and are explicitly unavailable. No attempt
is made to infer their groups from original templates.

Bounded mode writes no additional lineage. It finds the target's merge in the
validated semantic journal and replays its prefix against **only that target's
retained rows**, using immutable base assignments and any required predicate
attributes. This recovers exact retained membership across saved sessions,
nested merges and source-amplitude splits. It never opens the full source.

Every returned partition must be disjoint and exactly cover the live merged
unit's rows. Exposed membership arrays are read-only. Boundaries are checked
again when requested; no stale cached group can silently become a different
scientific unit.

## Display limits

At most 16 immediate parents are compared, with an explicit unavailable reason
above that limit. Existing per-view spike budgets are respected. Ordinary
correlograms use independent reproducible sampling when capped; regular thinning
is deliberately avoided because it can impose artificial short-lag structure.
Bounded correlograms sum only within retained contiguous windows and provide no
full-recording rate baseline. Full counts and rates for original or whole-merge
parents come from the manifest and semantic merge counts; split-derived full
counts are reported unavailable when they cannot be derived from those summaries.

Waveforms use only parent-member stored/raw spikes. If neither raw nor stored
waveforms are available, the view is explicitly labelled as template traces
(at most 100 per parent), weighted from that parent's exact membership. This
fallback changes waveform representation, never parent grouping. Empty evidence
parents remain in the correlogram matrix and legend; unavailable waveform or
feature groups are omitted from those plots without changing their colors.

No Unmerge command is implemented. Published documentation and changelog entries
are left to the integrating agent.

## Validation

Focused tests cover exact nested/split/undo/redo membership, read-only arrays,
ordinary save/reopen and stale-state refusal, provider spike identities, all
four comparison views, clean dirty-state/history/metadata/assignment invariants,
unchanged saved sidecar bytes, complete callback restoration, selection/mutation
exit, and bounded nested/predicate-split save/reload and journal invariants.

These tests run with the repository's headless Qt configuration. They establish
provider and GUI-state behavior, not physical OpenGL presentation. The integrating
agent owns the full application suite and final visual verification.
