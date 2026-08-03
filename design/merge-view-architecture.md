# Merge View architecture record

Status: implemented and automatically validated on the unreleased phy 2.2 branch;
manual dataset smoke testing and release acceptance remain

This document records the internal architecture and incremental refactor used to
implement the user behavior fixed in the
[Merge View workflow specification](merge-view-workflow.md). The workflow
specification is authoritative when this document discusses implementation
tradeoffs.

## 1. Scope

Merge View is a significant selection and curation workflow, not merely a third
table. It introduces:

- an explicit workflow mode;
- a third cluster role alongside Cluster and Similarity;
- an explicit presentation order derived from the two active roles;
- a fixed blue Similarity reference;
- stable cross-view colors independent of role and row-order changes;
- exact cancellation to an entry snapshot; and
- restoration of the complete workspace after undoing a committed merge.

The architectural goal is to support this workflow while improving the current
selection, action, and history boundaries. The refactor should be incremental and
should preserve public plugin APIs.

The following were out of scope for the manual-workflow implementation:

- Merge Propositions and proposition review states, now specified separately in
  [Merge Propositions](merge-propositions.md);
- a global application-state framework;
- a rewrite of scientific/OpenGL views;
- clustering-algorithm changes; and
- unrelated GUI modernization.

## 2. Pre-implementation architecture and constraints

### 2.1 Selection authority is indirect

`ClusterView` and `SimilarityView` each store selected row IDs. Their selection
events are logged by `TaskLogger`. `Supervisor.selected_clusters` and
`Supervisor.selected_similar` then reconstruct the latest selection by scanning
that task history.

Consequently, `TaskLogger` currently has several responsibilities:

- serializing actions and compatibility callbacks;
- retaining a diagnostic task log;
- serving indirectly as the authoritative selection store;
- reconstructing selection for undo and redo; and
- deciding wizard and post-action navigation.

Extending `TaskLogger.last_state()` with a third view would make selection state
more implicit and increase the number of callback-order dependencies.

### 2.2 Merge candidate transfers are multi-step callback sequences

Moving a candidate between Similarity and Merge requires changing both role
tables while preserving one effective selection and publishing only the final
presentation order.

Merge mode requires many related transitions: enter, cancel, add, remove,
reorder, commit, undo, and redo. Implementing each as a separate callback chain
would make intermediate invalid states observable and difficult to test.

### 2.3 Similarity reference and blue color are not explicit

`SimilarityView.reset()` uses the last Cluster selection as its reference. Table
and scientific-view colors are assigned from the position of a cluster in the
effective selected-ID list, where the first position is blue.

With several Cluster selections, the active Similarity reference is therefore
not necessarily blue. The target architecture makes the reference explicit and
guarantees that it occupies the blue presentation slot.

This is an intentional correction of the internal model. Characterization tests
must document the existing multi-selection behavior before changing it, and the
user-visible consequence must be reviewed during implementation.

### 2.4 Color slots are independent of presentation order

The authoritative selection state owns both an explicit `presentation_order`
and independent `color_slots`. Normal-mode presentation follows the visible
role tables; Merge ordering is modeled separately and takes precedence while
active. Normal color transitions depend on structured selection intent. In
Merge mode, existing bindings are retained across transfers, reordering,
temporary deselection, and reselection. Built-in scientific views resolve their
positional palette index through that mapping while retaining presentation
order for layout.

### 2.5 History lacks orchestration context

`GlobalHistory` records the data controllers participating in an action, but not
the selection or workflow context before and after the action. `TaskLogger`
currently reconstructs selection after undo and redo by inspecting its task log.

The low-level clustering and metadata controllers support a
`request_undo_state` event. That remains useful for controller-specific data, but
the Merge workspace is Supervisor-level workflow context and should ultimately
belong to the global curation-history entry.

### 2.6 UI action disabling is not sufficient enforcement

The `Actions` class can disable a Qt action, but snippets, plugins, and direct
method calls may execute the callback independently of the QAction enabled
state. Workflow restrictions must therefore be enforced at a common
Supervisor/controller boundary. Menu and shortcut enablement should reflect the
same policy rather than define it.

### 2.7 Selection delivery is debounced

Table row state changes synchronously, but selected-row events may be debounced.
The current authoritative state can therefore temporarily lag behind the visible
table. Entering Merge mode must not capture stale selections, and delayed events
from a previous mode must not mutate the new mode.

The target architecture separates synchronous state transitions from debounced
or coalesced scientific-view rendering.

### 2.8 Public compatibility matters

Plugins and application controllers use:

- `Supervisor.selected`;
- `Supervisor.selected_clusters`;
- `Supervisor.selected_similar`;
- `Supervisor.select()`;
- `emit('select', supervisor, cluster_ids)`; and
- explicit Supervisor clustering methods.

These boundaries should remain compatible while internal state ownership
changes.

## 3. Architectural goals and invariants

The refactor should establish the following invariants:

1. There is one authoritative curation-selection state.
2. Tables are projections of that state and sources of user intent, not
   independent domain authorities.
3. Similarity reference is an explicit cluster ID.
4. The reference occupies the blue presentation slot.
5. Presentation order follows visible Cluster and Similarity row order in
   Normal mode, with the reference first. In Merge mode it is Merge row order
   followed by visible Similarity selection order. Color slots are stable across
   workflow tables and scientific views independently of those order changes.
6. Moving a cluster between Similarity and Merge does not change membership,
   but emits a public selection update when it changes presentation order.
7. Related state changes are applied transactionally; observers see only valid
   before and after states.
8. Cancellation restores the exact entry snapshot.
9. Undoing a Merge-mode merge restores the exact pre-commit workspace.
10. Workspace edits do not enter the curation undo stack.
11. Public selection and plugin APIs remain compatible.
12. Selection transitions operate on cluster IDs and do no work proportional to
    every spike.

## 4. Proposed domain model

The exact module and class names may change during implementation. The important
decision is ownership and separation of responsibilities.

### 4.1 Workflow mode

```python
class WorkflowMode(Enum):
    NORMAL = "normal"
    MERGE = "merge"
```

No generic state-machine framework is needed. These two explicit modes and
validated transitions are sufficient for the manual workflow.

### 4.2 Authoritative selection state

```python
@dataclass(frozen=True)
class CurationSelectionState:
    mode: WorkflowMode
    cluster_ids: tuple[int, ...]
    similar_ids: tuple[int, ...]
    reference_id: int | None
    presentation_order: tuple[int, ...]
    color_slots: tuple[int | None, ...]
    merge: MergeSession | None
```

The state exposes derived values rather than storing duplicated lists where
possible:

```python
state.effective_ids
state.merge_ids
state.is_merge_mode
```

In Normal mode, effective membership is Cluster plus Similarity membership. In
Merge mode, it is Merge plus Similarity membership. `presentation_order` is the
ordered unique list emitted to scientific views. `color_slots` independently
stores explicit cluster-to-palette bindings, including released holes and
reserved inactive bindings.

### 4.3 Merge session

```python
@dataclass(frozen=True)
class MergeSession:
    reference_id: int
    ordered_ids: tuple[int, ...]
    entry_snapshot: NormalWorkflowSnapshot
```

`ordered_ids[0]` is always `reference_id`. The reference cannot be removed or
reordered. Reordering later candidates changes `ordered_ids` but not
`presentation_order`.

The entry snapshot contains the user state required by the workflow contract,
not arbitrary application state. It includes selections, reference,
presentation order, and any table filter, sort, scroll, or navigation state that
entering or editing Merge mode changes.

### 4.4 Selection change

```python
@dataclass(frozen=True)
class SelectionChange:
    before: CurationSelectionState
    after: CurationSelectionState
    roles_changed: bool
    presentation_changed: bool
    colors_changed: bool
    reference_changed: bool
    mode_changed: bool
```

This diff determines which observers need work:

- `roles_changed`: update Cluster, Similarity, and Merge projections;
- `presentation_changed` or `colors_changed`: emit the public `select` event;
- `reference_changed`: recompute Similarity candidates; and
- `mode_changed`: update action availability and enabled views.

Role-only transfers must not set `presentation_changed`.

### 4.5 Curation selection controller

The controller owns state and exposes domain operations rather than view
mutations:

```python
enter_merge_mode()
cancel_merge_mode()
add_to_merge(cluster_ids, insertion=None)
remove_from_merge(cluster_ids)
reorder_merge(cluster_id, insertion)
set_cluster_selection(cluster_ids)
apply_similarity_mutation(mutation)
clear_similarity_selection()
```

Each operation validates preconditions, calculates one new state, and publishes
one `SelectionChange` after the complete transition.

## 5. Ownership and event flow

The intended flow is:

```text
user gesture / action / plugin call
                 |
                 v
          Supervisor facade
                 |
                 v
   CurationSelectionController
       |                    |
       v                    v
table projections     effective selection
                            |
                            v
                 legacy public select event
                            |
                            v
                    scientific views
```

### 5.1 Tables emit intent

Cluster and Similarity tables may continue to use the generic Table selection
API, but their Supervisor adapters should interpret table events as proposed
state transitions. The controller accepts or rejects the intent and updates all
table projections under notification suppression.

Programmatic `Supervisor.select()` should use the same controller path.

### 5.2 Internal and public events are distinct

Internal state changes need richer information than the public plugin event.
They should use a dedicated internal event or direct observer interface carrying
`SelectionChange`.

The existing public event remains:

```python
emit('select', supervisor, list(state.presentation_order))
```

It is emitted only when effective presentation changes. Do not add new required
arguments to this event, because existing plugin callbacks may not accept them.

### 5.3 Synchronous state, coalesced rendering

Authoritative selection state changes synchronously with the accepted user
intent. Expensive scientific-view updates may remain debounced or coalesced.

Transitions should carry a generation or revision number, or otherwise cancel
stale pending table events, so an event initiated in Normal mode cannot arrive
later and corrupt Merge mode.

### 5.4 Transaction boundaries

Enter, cancel, merge completion, undo restoration, and redo restoration must each
be one transaction. Table projections may update internally during the
transaction, but no public selection event is emitted until the final state is
valid.

This replaces callback sequences that rely on `update_views=False`.

## 6. Actions, task sequencing, and history

### 6.1 Supervisor remains the public facade

`Supervisor` remains the public integration point for actions, plugins, views,
clustering, metadata, and selection properties. It delegates state transitions
to the new controller rather than accumulating additional mode-specific fields.

### 6.2 Central workflow policy

Action availability should derive from workflow state. The policy needs an
explicit matrix covering at least:

- merge;
- split;
- cluster metadata and group changes;
- Cluster and Similarity navigation;
- selection snippets;
- save;
- undo and redo; and
- plugin/programmatic selection requests.

The policy is enforced at the Supervisor/controller command boundary. QAction
enabled state, disabled table interaction, and status messages mirror that
decision.

The exact availability of non-merge curation actions during Merge mode should be
recorded in the implementation plan before code is merged. The safe default is
to reject operations that could invalidate workspace cluster IDs, while allowing
Similarity exploration, save, merge, undo, and redo where their semantics are
defined.

### 6.3 Decompose `TaskLogger` incrementally

Do not rewrite `TaskLogger` and selection simultaneously. First remove selection
authority from it. It may temporarily continue to:

- serialize callback-compatible actions;
- record diagnostic tasks; and
- initiate post-action work.

After Normal-mode selection is controlled and tested, split the remaining roles
conceptually into:

- `ActionRunner`: serialization, busy-state coordination, and callbacks; and
- `PostActionPolicy`: wizard navigation and selections following merge, split,
  move, undo, and redo.

Post-action decisions receive explicit before-state and action results instead of
recovering them by scanning a task log.

### 6.4 Contextual history entries

Global curation history should evolve from a tuple of controllers to an action
entry:

```python
@dataclass
class CurationHistoryEntry:
    controllers: tuple[object, ...]
    description: str
    selection_before: CurationSelectionSnapshot
    selection_after: CurationSelectionSnapshot
    workflow_context: object | None = None
```

The low-level clustering and metadata histories continue to own reversible data
changes. The global entry coordinates those controllers with selection/workflow
restoration.

For a Merge-mode merge:

1. capture the complete pre-commit Merge state;
2. execute the clustering merge;
3. capture the resulting Normal-mode state;
4. store both on the global action entry;
5. on undo, undo controllers and restore `selection_before` transactionally; and
6. on redo, redo controllers and restore `selection_after` transactionally.

The existing `request_undo_state` mechanism may be used as a compatibility step,
but the final ownership of Merge workflow context belongs to the global curation
action, not the `Clustering` model.

## 7. View integration

### 7.1 Cluster View

During Merge mode, Cluster View remains visible but disabled. Its Qt interaction,
selection actions, snippets, and programmatic workflow mutations must all be
consistent with the central action policy. A visible overlay or dock status
explains why it is disabled.

### 7.2 Similarity View

Similarity View receives an explicit reference ID and a set of Merge IDs to
exclude from its candidate rows. Its selection remains an ordinary ordered
selection contributing to effective Merge-mode membership.

Backspace should directly clear Similarity selection through the controller. It
should no longer depend on reselecting Cluster View.

### 7.3 Merge View

Merge View is a projection of `MergeSession.ordered_ids`, not an independent
selection owner. Every row is included. It has no selected/unselected scientific
state.

It may maintain lightweight local hover or drag state, but clicking a row does
not change effective selection. The reference row is fixed and visually blue.

### 7.4 Drag-and-drop

Generic native-table infrastructure should provide:

- a cluster-ID-only MIME payload;
- single- and multi-row drag initiation;
- insertion indicators;
- accepted-source and accepted-target policies; and
- drag cancellation and duplicate protection.

Domain views emit transfer or reorder intents. They do not directly mutate the
other table.

During Merge mode, the relevant transfers are Similarity-to-Merge,
Merge-to-Similarity, and internal Merge reorder. Cluster View is disabled. The
first implementation should establish the complete workflow without
drag-and-drop, then add this reusable layer.

### 7.5 View closing and lifecycle

Closing Merge View is a cancel intent. Cancellation must complete before the view
is removed or hidden. Re-entering Merge mode must be able to recreate or reveal
the view without retaining stale local state.

Application shutdown must not accidentally save the transient empty Cluster
selection produced by Merge mode as the next Normal-mode selection. Either
cancel to the entry snapshot before saving GUI state or deliberately save the
entry snapshot as the Normal-mode state.

Busy-state and test-flush logic should use a small registry of participating
selection views/controllers rather than hard-code only Cluster and Similarity
tables.

## 8. Compatibility strategy

The refactor preserves these public behaviors unless separately approved:

- `Supervisor.selected` returns the effective presentation order;
- `Supervisor.selected_clusters` and `selected_similar` remain available;
- `Supervisor.select()` continues to select clusters in Normal mode;
- the public `select` event retains its current positional argument shape;
- existing scientific views continue assigning colors by selection-list index;
  and
- ordinary merge, split, move, wizard, undo, and redo behavior remains covered
  by characterization tests.

A new `selected_merge` or read-only workflow-state property may be added, but
plugins should not be required to use it for existing workflows.

Direct calls made during Merge mode require a documented policy. Rejection must
be explicit and logged; silently modifying disabled Cluster View state is not
acceptable.

## 9. Incremental migration plan

Each phase should leave the repository testable and avoid combining broad
behavioral changes with mechanical moves.

### Planned commit sequence

Implementation is organized as twelve reviewable commits. A commit may be split
if its diff becomes difficult to review, but independent phases should not be
squashed together merely to preserve the count.

1. `test: characterize curation selection contracts`
2. `refactor: add curation selection state model`
3. `refactor: shadow supervisor selection state`
4. `refactor: make curation selection authoritative`
5. `refactor: separate reference and presentation order`
6. `refactor: remove selection state from task history`
7. `refactor: add contextual curation history`
8. `feat: add merge session and mode lifecycle`
9. `feat: add merge candidate interactions`
10. `feat: restore merge sessions through history`
11. `feat: add cluster table drag and drop`
12. `docs: finalize merge view workflow`

Commits 1-7 form the architectural foundation (Milestone 1). Commits 8-10
provide the complete manual workflow without drag-and-drop (Milestone 2).
Commits 11-12 add drag-and-drop and final cleanup/documentation (Milestone 3).

The initial Merge-mode action policy is:

- disable split, group/metadata changes, Cluster navigation, and Cluster
  selection;
- allow Similarity navigation, filtering, sorting, Ctrl+Space, Backspace, `V`,
  `G`, and save;
- reject unsafe direct or plugin calls explicitly without partially mutating the
  workspace;
- do not let an uncommitted Merge session undo an earlier curation action;
- after undoing a Merge-mode merge, allow redo to reapply it; and
- truncate that redo branch normally if the restored workspace is edited and a
  different curation action is committed.

Cancellation and shutdown restore or persist the Normal-mode entry snapshot,
never the transient empty Cluster selection. The snapshot includes selections,
reference, ordering, filter, sort, and navigation state. Pixel-perfect scroll
restoration is best effort where Qt exposes a reliable value.

### Phase 0: characterization

- Add tests for selection order, effective selection, multi-Cluster Similarity
  reference, positional colors, Ctrl+Space, Backspace, merge follow-up,
  undo/redo selection restoration, debouncing, and plugin-facing events.
- Record exact event counts for transfers, reorders, and merges.

### Phase 1: state model in observation mode

- Add immutable state and change types.
- Derive shadow state from existing Normal-mode events.
- Assert agreement with current `TaskLogger`-derived properties in tests.
- Do not change user-visible behavior yet.

### Phase 2: authoritative Normal-mode selection

- Route table and programmatic selection through the controller.
- Make Supervisor selection properties read from the controller.
- Preserve the legacy public event.
- Remove selection reconstruction from `TaskLogger.last_state()`.

### Phase 3: explicit reference and presentation order

- Make Similarity reference explicit.
- Establish the reviewed blue-reference invariant.
- Derive Merge-mode presentation order from table-role order.
- Update characterization tests for the intentionally approved behavior change.

### Phase 4: transactional actions and contextual history

- Add transaction and state-diff publication.
- Introduce contextual global-history entries.
- Restore selection from snapshots on ordinary merge/split/move undo and redo.
- Retain the existing task queue until behavior is equivalent.

### Phase 5: Merge mode without drag-and-drop

- Add `MergeSession` and mode transitions.
- Add Merge View, `V`, Ctrl+right-click transfers, Backspace behavior, mode
  indication, cancellation, and `G` semantics.
- Add exact cancel and merge undo/redo restoration tests.

### Phase 6: drag-and-drop

- Add reusable native-table drag support.
- Connect it to controller transfer and reorder intents.
- Verify multi-row behavior and presentation updates across platforms.

### Phase 7: TaskLogger decomposition and cleanup

- Extract post-action policy from serialized action execution.
- Remove obsolete selection-history parsing and `update_views=False` paths.
- Retain diagnostic action logging where useful.

### Phase 8: documentation and release verification

- Update user documentation, shortcuts, and the unreleased changelog.
- Run focused tests throughout development.
- Before handoff, run `make lint`, `make format-check`, `make test-full`, and
  `make doc-check` as required by repository guidance.

## 10. Testing and performance strategy

### 10.1 Pure state-transition tests

Most mode behavior should be testable without Qt:

- valid and invalid entry;
- fixed reference;
- transfer and reorder;
- mode-dependent presentation order;
- exact cancellation;
- effective membership and merge target; and
- before/after snapshot restoration.

### 10.2 Supervisor integration tests

Cover:

- legacy selection properties and events;
- Similarity reference updates;
- action policy enforcement;
- single public event per effective change;
- no public event for role-only transfers;
- merge failures;
- ordinary and Merge-mode undo/redo; and
- plugin-compatible direct calls.

### 10.3 Qt tests

Cover:

- disabled Cluster View interaction;
- mode indicator and pending count;
- Ctrl+right-click;
- close-to-cancel;
- drag-and-drop and insertion order;
- reference-row immobility; and
- view recreation and shutdown state.

### 10.4 Safety regressions

Merge, selection, undo, saved assignments, and cross-view consistency are
safety-sensitive. Tests must verify cluster IDs, descendants, resulting
assignments, metadata inheritance, undo/redo symmetry, and save output after a
Merge-mode merge.

### 10.5 Performance

- State diffs operate on the small selected-ID collections.
- Role transfers do not rescan spikes or rebuild all Cluster rows.
- Similarity candidate updates occur only when reference or exclusions change.
- Scientific views are not notified when effective presentation is unchanged.
- Drag payloads contain cluster IDs, never spike arrays.

## 11. Alternatives considered

### Extend `TaskLogger.last_state()`

This is the smallest initial change but leaves state implicit, adds a third view
to history scanning, and multiplies callback-order special cases. It is not the
recommended long-term direction.

### Add Merge-specific fields directly to `Supervisor`

This would provide an authoritative workspace but leave Cluster and Similarity
selection authority in tables and TaskLogger. Transitions would still need to
coordinate multiple owners. It may be useful as a prototype but should not be
the production architecture.

### Replace the entire app with a global reactive store

This could unify all state eventually, but the migration surface includes every
view and plugin. It is disproportionate to the feature and too risky for the
curation path.

### Derive colors exclusively from presentation order

This initially minimized the migration surface, but it recolored existing
clusters whenever inserting a newly selected row earlier in presentation order
or transferring a Merge candidate between roles. The explicit color-slot order
avoids that cross-view inconsistency while leaving the public selection payload
unchanged.

### Store Merge UI context inside `Clustering`

The existing undo-state hook makes this possible, but the workspace is not spike
assignment state. Contextual global-history entries preserve the domain boundary
and generalize to future workflows.

## 12. Deferred decisions and work

The following are not blockers for writing this architecture but must be fixed
before their implementation phase:

- the exact availability matrix for non-merge curation actions in Merge mode;
- whether rejected programmatic actions return a structured result, raise, or
  only warn while preserving current public compatibility;
- the reviewed Normal-mode behavior when multiple Cluster rows are selected and
  a new reference becomes blue; and
- whether the first contextual-history implementation uses the existing
  `request_undo_state` hook as a transition step.

TaskLogger no longer owns selection state, but its optional structural split into
separate action-runner and post-action-policy classes remains deferred cleanup.
Merge Propositions and their persistence are specified in
[Merge Propositions](merge-propositions.md).

## 13. Handoff for future agents

Before modifying code:

1. Read the repository `AGENTS.md`.
2. Read `design/merge-view-workflow.md` completely.
3. Read this document completely.
4. Inspect the current working tree and preserve unrelated user changes.
5. Start with Phase 0 characterization tests; do not begin with Merge View UI.

Primary existing integration points are:

- `phy/cluster/supervisor.py`: `TaskLogger`, `ClusterView`, `SimilarityView`,
  `ActionCreator`, and `Supervisor`;
- `phy/cluster/_history.py`: local and global history;
- `phy/cluster/clustering.py`: merge/split undo state and cluster updates;
- `phy/gui/widgets.py`: native Table selection and future drag support;
- `phy/gui/actions.py`: Qt actions, snippets, and enablement;
- `phy/utils/color.py`: positional selected-cluster colors;
- `phy/cluster/tests/test_supervisor.py`: current selection and action
  characterization; and
- `phy/gui/tests/test_widgets.py`: native Table behavior.

Do not silently weaken these central invariants to simplify implementation:

- the reference is blue and fixed in Merge View;
- all entry selections transfer into Merge View;
- Merge-mode transfers and reorders publish their resulting presentation order;
- cancellation restores the exact entry state;
- `G` has one user-facing meaning in each visibly distinct mode; and
- undoing a Merge-mode merge restores the complete pre-commit workspace.

If implementation reveals a conflict with the workflow specification, update the
design through explicit maintainer discussion rather than allowing code details
to redefine the workflow.
