# Amplitude-threshold splitting implementation plan

Status: implemented on the unreleased phy 2.2 branch; final integration and manual validation pending

## 1. Goal

Add a fast way to split the low-amplitude part of exactly one selected cluster.
The curator places a horizontal threshold in Amplitude View, previews the same
spike subset in Amplitude View and Waveform View, and presses `K` to perform the
ordinary phy split.

This is a curation-integrity feature. The preview and committed spike IDs must
be derived from the same amplitude definition, and the committed selection must
be evaluated over every eligible spike rather than only the displayed sample.

## 2. Locked user-visible behavior

These decisions are implementation inputs, not questions for delegated agents.

1. The interaction is available only when exactly one cluster is selected and
   Merge mode is inactive.
2. `Alt`-right-drag in Amplitude View creates or moves a horizontal threshold.
   Plain right-drag remains zoom.
3. The existing `Alt` time-selection gesture must respond only to left-click so
   the two interactions do not overlap.
4. Spikes with finite amplitude strictly below the threshold are the pending
   split subset. Equality remains in the upper group.
5. Below-threshold Amplitude View points and corresponding individual waveform
   traces use one dedicated preview color. Above-threshold spikes retain the
   selected cluster color; background spikes are unchanged.
6. Releasing the mouse leaves the preview active. Another `Alt`-right-drag
   adjusts it.
7. `Control`-right-click clears both the Amplitude View lasso and threshold.
   A view-menu action named `Clear amplitude split threshold` provides a
   discoverable alternative.
8. Pressing `K` evaluates all eligible spikes in the cluster with the current
   amplitude type and context, then calls the existing split path.
9. A threshold selecting zero spikes or the entire cluster does not call the
   clustering model. It remains visible so the curator can adjust it, and a
   status message explains why the split was rejected.
10. The threshold is transient. It is never saved in GUI state and is cleared
    after a successful request, selection change, cluster update, amplitude-type
    change, or a channel/PC change that alters the amplitude definition.
11. Only individual waveform mode receives per-spike highlighting in the first
    version. Mean and template waveform modes remain unchanged.
12. Only the most recently edited built-in split selection is active. Starting
    a threshold preview clears built-in lassos; starting a lasso clears the
    threshold and other built-in lassos. Existing third-party `request_split`
    listeners remain compatible.

The initial version selects the lower side only. Inverting the selected side,
selecting an amplitude band, coloring histogram bins, and displaying separate
above/below mean waveforms are explicitly deferred.

## 3. Current implementation boundaries

- `phy/cluster/views/amplitude.py` already owns amplitude positions, spike IDs,
  per-point colors, histograms, time selection, and lasso splitting.
- `LassoMixin.on_request_split()` in `phy/cluster/views/base.py` already reloads
  all spikes for an exact split.
- `Supervisor.split()` in `phy/cluster/supervisor.py` gathers
  `request_split` results and invokes the existing clustering/history path.
- `BaseController._amplitude_getter()` in `phy/apps/base.py` is the current
  authority for choosing spike IDs and evaluating the active amplitude type,
  channel, and PC.
- `WaveformMixin._get_waveforms_with_n_spikes()` chooses waveform spike IDs but
  does not currently return those IDs.
- `WaveformView` currently assigns one cluster color to every trace, although
  `PlotVisual` accepts one color per signal.

Do not add a second split implementation to `Clustering`, write spike-cluster
arrays directly, or make the displayed amplitude sample authoritative.

## 4. State and event contract

### 4.1 Amplitude View transient state

`AmplitudeView` should own only view-level preview state:

```python
split_threshold: float | None
split_preview_color: tuple[float, float, float, float]
```

It may cache its current displayed bunches for recoloring. It must not cache all
cluster amplitudes merely to support dragging.

The view constructor may accept an optional eligibility predicate supplied by
the controller. Standalone views default to checking their local cluster count;
the application predicate additionally checks the Supervisor's complete
selection and Merge-mode state. This prevents a truncated view selection from
becoming the authority for whether the gesture is allowed.

### 4.2 Preview context

Every threshold-preview update must identify:

```text
cluster_id
amplitudes_type
threshold (or None when clearing)
```

The controller remains responsible for resolving the amplitude type to the
same channel IDs, selected channel, selected PC, and first cluster used by the
Amplitude View. Do not duplicate that context-selection logic in Waveform View.

Use a narrowly named event such as `amplitude_split_preview_changed`. The event
is transient UI coordination and must not enter curation history or GUI state.

### 4.3 Exact versus sampled work

- During mouse movement, recolor only points and waveforms already displayed.
- Resolve amplitudes only for the displayed waveform spike IDs. Cache those
  amplitude values for the unchanged preview context so moving the threshold
  performs comparisons, not repeated data loads.
- On `K`, call the existing amplitude provider with `load_all=True` exactly
  once, filter finite amplitudes with `< threshold`, and validate the result.

### 4.4 Exclusive built-in split preview

Add a small shared coordination layer for built-in lasso and threshold views:

- a view announces when it activates a split selection;
- other built-in split-capable views clear their transient selection;
- clearing is visual/state cleanup only and never changes cluster assignments;
- the ordinary `request_split` event remains the commit boundary.

Implement this in the split-view mixins or a small helper in
`phy/cluster/views/base.py`; do not put transient view state in
`CurationSelectionState`. Feature View has a custom split implementation and
must join the same coordination contract explicitly.

## 5. Delegation map

Each work package is intentionally bounded enough for a cheaper coding model at
low reasoning effort. Agents must read `AGENTS.md` and this document completely
before editing. They must inspect `git status`, preserve concurrent work, stage
only listed files, and report any unexpected overlap instead of rewriting it.

The integration owner should assign one package per branch or worktree. Packages
B and C may run in parallel after A. Package D depends on both B and C. Package
E depends on D. Package F is the final integration audit.

```text
A: split-preview coordination
          |\
          | +--> B: Amplitude View threshold core --+
          |                                       |
          +----> C: Waveform identity/highlight --+--> D: controller bridge
                                                       |
                                                       v
                                               E: integration/docs
                                                       |
                                                       v
                                                  F: final audit
```

No package may edit `phy/cluster/_selection.py` or
`phy/cluster/supervisor.py`; those files may contain concurrent selection work,
and this feature does not require changes to either file.

## 6. Work package A: exclusive built-in split previews

Suggested agent: cheaper coding model, low reasoning.

Dependencies: none.

Owned files:

- `phy/cluster/views/base.py`
- `phy/cluster/views/feature.py`
- `phy/cluster/views/tests/test_base.py`
- `phy/cluster/views/tests/test_feature.py`
- `phy/cluster/views/tests/test_scatter.py` only if needed

Tasks:

1. Introduce the smallest shared mixin/helper that can announce activation and
   clear a transient split selection.
2. Make non-empty lasso updates activate their owning built-in view.
3. When another built-in view activates, clear the current view's lasso.
4. Include Feature View despite its custom `on_request_split()` method.
5. Ensure connection cleanup follows existing view lifecycle patterns and does
   not retain closed views.
6. Preserve standalone view behavior and the public `request_split` event.

Acceptance tests:

- Drawing a lasso in view A and then view B clears A.
- Redrawing in A clears B.
- The latest lasso still returns the expected unique spike IDs.
- Closing either view leaves no callback into the closed canvas.
- A standalone lasso view continues to split without a Supervisor.

Run:

```bash
uv run pytest phy/cluster/views/tests/test_base.py
uv run pytest phy/cluster/views/tests/test_feature.py
uv run pytest phy/cluster/views/tests/test_scatter.py
```

Suggested commit: `refactor: coordinate transient split selections`

## 7. Work package B: Amplitude View threshold core

Suggested agent: cheaper coding model, low reasoning.

Dependencies: package A.

Owned files:

- `phy/cluster/views/amplitude.py`
- `phy/cluster/views/tests/test_amplitude.py`

Tasks:

1. Add transient threshold state and a horizontal line visual. Keep the line in
   amplitude data coordinates so pan and zoom do not alter its meaning.
2. Restrict `Alt` time selection to left-click.
3. Implement `Alt`-right press/move/release. Convert the pointer Y coordinate
   through the same NDC-to-data transform used by the plotted amplitudes and
   clamp only if existing view bounds require it.
4. Require exactly one selected cluster. On invalid activation, leave state
   unchanged and log a concise warning/status message.
   Expose an optional eligibility predicate for package D to enforce the full
   application selection and Merge-mode condition without importing Supervisor
   state into the view.
5. Recolor displayed selected-cluster points below the threshold with a
   per-point color array. Never recolor background points.
6. Emit the preview event after threshold changes and when it is cleared.
7. Make threshold activation participate in package A's exclusive split-preview
   contract. `Control`-right-click clears both lasso and threshold.
8. Override `on_request_split()`:
   - delegate to the lasso implementation when no threshold exists;
   - otherwise reload the single cluster with `load_all=True` once;
   - ignore non-finite amplitudes;
   - select strict `< threshold` spike IDs;
   - reject empty and whole-cluster results without clearing the threshold;
   - return unique `int64` spike IDs and clear after a valid request.
9. Clear the preview on selection, cluster, and amplitude-type changes. Do not
   persist it in `state_attrs` or `local_state_attrs`.
10. Add the view action and shortcut metadata needed for help generation.

Acceptance tests:

- Gesture-to-data conversion remains correct after pan/zoom.
- `Alt`-right interaction does not emit `select_time`; `Alt`-left still does.
- Background colors remain unchanged.
- Multi-cluster activation is rejected.
- Exact request uses spikes absent from the displayed sample.
- Equality, NaN, empty-side, whole-cluster, and valid-side cases are explicit.
- Threshold state clears on every context change listed in section 2.
- Existing lasso splitting tests still pass.

Run:

```bash
uv run pytest phy/cluster/views/tests/test_amplitude.py
```

Suggested commit: `feat: preview amplitude threshold splits`

## 8. Work package C: waveform spike identity and recoloring

Suggested agent: cheaper coding model, low reasoning.

Dependencies: package A. May run in parallel with B.

Owned files:

- `phy/apps/base.py`, limited to waveform-provider changes
- `phy/cluster/views/waveform.py`
- `phy/cluster/views/tests/test_waveform.py`
- `phy/apps/tests/test_base.py`, limited to waveform-provider tests

Tasks:

1. Include the sampled `spike_ids` in the individual-waveform `Bunch` returned
   by `_get_waveforms_with_n_spikes()`.
2. Keep mean/template waveform contracts valid. An aggregated waveform must not
   pretend that many source IDs map one-to-one to its single trace.
3. Refactor `WaveformView.plot()` only as much as needed to retain its current
   displayed bunches and rerender colors without calling the waveform provider
   again.
4. Add a method accepting highlighted spike IDs or an empty value. It must:
   - operate only in individual waveform mode;
   - intersect against each bunch's displayed spike IDs;
   - create one color per spike and repeat it across that spike's channels in
     the exact signal order produced by the current transpose/reshape;
   - preserve masks, box indices, overlap behavior, axes, and base colors.
5. Clear highlighted IDs on selection/cluster changes and when leaving
   individual waveform mode.
6. Avoid allocations proportional to the complete recording or cluster.

Acceptance tests:

- Waveform bunch spike IDs exactly match the sampled data's first dimension.
- Only matching traces change color, on every channel belonging to the spike.
- Nonmatching traces retain their cluster color.
- Highlight updates do not invoke the waveform provider again.
- Multi-cluster rendering, overlap, masks, channel labels, and mean-waveform
  toggling retain existing behavior.
- Missing `spike_ids` from a custom waveform provider disables highlighting
  gracefully rather than guessing.

Run:

```bash
uv run pytest phy/cluster/views/tests/test_waveform.py
uv run pytest phy/apps/tests/test_base.py -k waveform
```

Suggested commit: `feat: support transient waveform spike highlights`

## 9. Work package D: shared amplitude resolver and controller bridge

Suggested agent: cheaper coding model, low reasoning, with the merged outputs of
B and C available.

Dependencies: packages B and C.

Owned files:

- `phy/apps/base.py`
- `phy/apps/tests/test_base.py`

Tasks:

1. Extract a private helper from `_amplitude_getter()` that evaluates a supplied
   spike-ID array using one named amplitude type and the canonical first-cluster,
   channel IDs, selected channel, and selected PC context.
2. Make `_amplitude_getter()` use that helper so Amplitude View plotting,
   threshold commit, and waveform preview cannot drift semantically.
3. In each Waveform View created by the controller, listen for amplitude split
   preview changes from the matching controller/view only.
4. Supply Amplitude View's eligibility predicate from the complete Supervisor
   state: exactly one selected cluster and Normal mode. Clear any active preview
   when that predicate becomes false.
5. For an active preview matching the Waveform View's sole selected cluster:
   obtain only its displayed individual-waveform spike IDs, resolve their
   amplitudes through the shared helper, cache those values for the unchanged
   `(cluster, amplitude type, channel context, PC context, spike IDs)` key, and
   pass the below-threshold IDs to Waveform View.
6. A threshold-only change must reuse the cached values. Selection, amplitude
   type, channel, PC, waveform IDs, filter changes, or cluster updates must
   invalidate them.
7. A clear/mismatched preview, closed view, unavailable amplitude data, or
   non-individual waveform mode must clear highlighting without raising.
8. Disconnect every added callback when either relevant view closes.

Acceptance tests:

- Feature and template amplitude previews use the same values as Amplitude View.
- Waveform IDs not present in the Amplitude View display sample are still
  classified correctly.
- Repeated threshold movement performs no repeated amplitude-provider load for
  unchanged context.
- Channel and PC changes invalidate both threshold semantics and preview cache.
- Events from another controller or Amplitude View do not affect this view.
- Closing and reopening either view does not duplicate callbacks.

Run:

```bash
uv run pytest phy/apps/tests/test_base.py -k 'amplitude or waveform or split'
uv run pytest phy/cluster/views/tests/test_amplitude.py
uv run pytest phy/cluster/views/tests/test_waveform.py
```

Suggested commit: `feat: link amplitude split previews to waveforms`

## 10. Work package E: application regression and documentation

Suggested agent: cheaper coding model, low reasoning.

Dependencies: package D.

Owned files:

- `phy/apps/tests/test_base.py`
- `phy/apps/template/tests/test_gui.py` if the existing fixture is suitable
- `docs/visualization.md`
- `docs/clustering.md`
- `docs/changelog.md`
- generated documentation changed by `make doc-check`

Tasks:

1. Add one end-to-end controller regression using deterministic spike IDs and
   amplitudes: select one cluster, set a threshold, verify both previews, press
   `K`, and verify the two new clusters contain exactly the expected spikes.
2. Verify undo restores the original assignments and clears transient preview
   state. If practical in the fixture, verify redo as well.
3. Verify a displayed waveform spike missing from the Amplitude View sample is
   highlighted from its actual amplitude.
4. Document the gesture, single-cluster requirement, lower-side semantics,
   clear action, `K` commit, individual-waveform limitation, and exact-all-spike
   commit behavior.
5. Add the user-visible feature to the unreleased changelog.
6. Regenerate shortcut/API documentation through the repository command; do not
   hand-edit generated sections unless the generator requires source changes.

Run:

```bash
uv run pytest phy/apps/tests/test_base.py -k split
uv run pytest phy/apps/template/tests/test_gui.py -k amplitude
make doc-check
```

Suggested commit: `docs: cover amplitude threshold splitting`

## 11. Work package F: integration and safety audit

Suggested owner: integration agent, medium reasoning. This is review and repair,
not a redesign package.

Dependencies: package E.

Owned files: only files already touched by packages A-E, and only for necessary
integration fixes. Do not absorb unrelated working-tree changes.

Checklist:

1. Review the complete diff against the locked behavior in section 2.
2. Confirm preview and commit share the canonical amplitude resolver.
3. Confirm no interactive drag path requests all cluster spikes.
4. Confirm `K` evaluates all eligible spikes exactly once.
5. Confirm zero/all/NaN cases cannot mutate clustering.
6. Confirm a successful split, undo, redo, save, and reload retain ordinary phy
   clustering semantics.
7. Confirm built-in stale lassos cannot be silently unioned with the threshold.
8. Confirm callbacks and OpenGL visuals are cleaned up when views close.
9. Inspect allocations for dependence on displayed spikes only during preview.
10. Inspect the index before every commit and exclude pre-existing changes.

Final validation:

```bash
uv run pytest phy/cluster/views/tests/test_base.py
uv run pytest phy/cluster/views/tests/test_feature.py
uv run pytest phy/cluster/views/tests/test_scatter.py
uv run pytest phy/cluster/views/tests/test_amplitude.py
uv run pytest phy/cluster/views/tests/test_waveform.py
uv run pytest phy/apps/tests/test_base.py
uv run pytest phy/apps/template/tests/test_gui.py
make lint
make format-check
make doc-check
make test-full
```

Because this changes GUI interaction, split selection, and cross-view spike
identity, passing only unit tests is not sufficient. The integration owner must
also perform a manual smoke test on a cluster larger than both display budgets:

1. Place and move the threshold while zoomed.
2. Confirm Amplitude and Waveform previews agree.
3. Commit and inspect both descendants.
4. Undo and redo.
5. Save, reopen, and verify the saved cluster assignments.

## 12. Handoff template for every delegated package

Each agent should return:

```text
Package:
Commit:
Files changed:
Behavior implemented:
Tests run and results:
Known limitations or follow-up:
Unexpected pre-existing changes left untouched:
```

An agent must not claim completion if required tests were skipped because of an
environment failure. Report the exact failure and leave the package for the
integration owner to verify.
