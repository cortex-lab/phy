# Long-recording curation in Phy 2.x: Phy-side design contract

**Status:** implementation present; acceptance validation is ongoing. This
specification is for Phy-side behavior; it does not authorize changes to the
AIND/Code Ocean export pipeline. The current branch implements the bounded
workflow behind an explicit manifest while retaining standard behavior for
ordinary datasets. See the [integration plan](large-recording-integration-plan.md)
and [validation record](large-recording-validation.md) for implementation scope,
checks, and remaining acceptance work.

The [integration audit and plan](large-recording-integration-plan.md) refines
this proposal with assignment revisions, transactional persistence, capability
gates and ordered commit acceptance criteria for one implementation PR. Its enablement gates take precedence
over the original implementation sequence in Section 9.

## 1. Objective and boundaries

Curate chronic Neuropixels sorting datasets of roughly 1000 hours without opening hundreds
of millions or billions of spikes in the existing in-memory Phy clustering model.
The workstation downloads the **complete sorter output once**; downloading the
complete raw voltage recording is **not required**. All preparation, interactive
curation, full-data replay, and regeneration run locally:

```text
complete, immutable sorting (local disk)
    |  prepare-large: streamed/batch
    v
bounded, temporally distributed Phy dataset + full-duration summaries
    |  existing Template GUI + Unit Stability
    v
semantic curation actions (not only sampled row assignments)
    |  apply-curation: streamed/batch against complete sorting
    v
new complete curated sorting (original untouched)
    |  prepare-large again, potentially different evidence windows
    +-----------------------> new bounded GUI dataset
```

The **complete spike-level sorting stays outside the interactive GUI**.
The bounded dataset must reproduce useful current Phy workflows rather than
pretend to be a statistically representative replacement for every spike.
Full-duration plots and rates come from separate complete-sorting summaries.

Out of scope: cloud services, rewriting phylib for out-of-core clustering,
longitudinal unit-identity graphs (phy3), automatic change-point detection, and
implementing upstream waveform extraction in this repository.

## 2. The three index spaces (essential contract)

All index values below are **zero-based row positions**, not timestamps, sample
numbers, SpikeInterface unit IDs, or stable IDs assigned by a sorter.

| Symbol | Meaning | Where it lives |
| --- | --- | --- |
| `g` | Full-sorting spike row index, in immutable source order | Original `spike_times.npy` / `spike_clusters.npy` |
| `b` | Bounded spike row index used by the GUI | Bounded `spike_times.npy` / `spike_clusters.npy` |
| `w` | Row in the stored waveform/channel tensors | `_phy_spikes_subset.waveforms.npy` / `.channels.npy` |

**Canonical mapping:** the bounded package writes
`_phy_spike_global_ids.npy`, a one-dimensional strictly increasing `int64`
array of length `N_bounded`. It is `G[b] = g`. The underscore prefix is
intentional: phylib treats ordinary `spike_*.npy` files as per-spike
attributes, so `spike_global_ids.npy` would be needlessly interpreted as one.

**Dataset-local waveform IDs:** `_phy_spikes_subset.spikes.npy[w] = b` in the
bounded package. In the *upstream full export*, the same filename instead
contains `g` because that file accompanies the full sorting. The file **never**
means an intrinsic immutable/global ID; it indexes whichever Phy dataset is in
its directory.

Example:

```text
full indices selected by prepare-large, G:
  [14, 19, 27, 117, 203, 839]
bounded rows b:
  [ 0,  1,  2,   3,   4,   5]
upstream waveform full indices:
  [14, 117, 839, 1902]
intersection with G:
  [14, 117, 839]
bounded _phy_spikes_subset.spikes.npy:
  [ 0,   3,   5]
```

Recover full source row numbers for waveforms by composition:

```python
G = np.load('_phy_spike_global_ids.npy', mmap_mode='r')
B = np.load('_phy_spikes_subset.spikes.npy', mmap_mode='r')
full_waveform_rows = G[B]  # [14, 117, 839]
```

`prepare-large` must preserve source row order, subset all per-spike arrays
consistently, retain absolute spike sample times, and remap matching waveform
rows **along with the corresponding channel/waveform data**. A sorted
intersection / `np.searchsorted` implementation needs no billion-element
inverse lookup array. Sorting by sample time is not a substitute for preserving
row identity; use the source export's exact row order (including ties).

## 3. Source and optional upstream waveform inputs

The source is a locally accessible standard Phy/SpikeInterface export (or an
adapter yielding equivalent arrays). The original spike index `g` refers to
**the exact exported row ordering**, not an assumed row in an unrelated
SpikeInterface spike vector. Source arrays and their provenance must be
validated before processing; never infer a row mapping from spike timestamps.

The existing, unmodified Phy waveform files are:

| Filename | Meaning |
| --- | --- |
| `_phy_spikes_subset.spikes.npy` | `int64[N_waveforms]`: source full spike row indices, sorted/unique |
| `_phy_spikes_subset.channels.npy` | `int32[N_waveforms, C_sparse]`: model channel indices, with `-1` padding; `channel_map.npy` maps these to raw-file columns |
| `_phy_spikes_subset.waveforms.npy` | numeric `[N_waveforms, N_samples, C_sparse]`: aligned waveform data, zero on padded channels |

The upstream AIND pipeline **may later** supply the following proposed files;
these names/schema are *not yet finalized with its developers*:

| Optional file | Initial proposal |
| --- | --- |
| `_phy_spikes_subset.window_bounds.npy` | `int64[W, 2]` recording sample intervals, `[start, end)` |
| `_phy_spikes_subset.window_types.npy` | `uint8[W]`: `0` survey, `1` rescue |
| `_phy_spikes_subset.window_strata.npy` | `int32[W]`: survey stratum index, `-1` for rescue |
| `_phy_spikes_subset.manifest.json` | Small versioned scalar metadata: source identity, time base, extractor geometry, parameters, seed, array semantics |

There are deliberately **no structured NumPy dtypes**, mixed-purpose `N x 5`
matrices, or massive JSON window arrays. The upstream manifest is useful for
planning downstream windows but is not required for fixture-based development.
The presence of a window denotes *potential* waveform coverage, not one stored
waveform for every spike within that interval: per-unit waveform caps still
apply. The three established waveform files remain authoritative for which
individual spikes actually have stored waveforms.

In the absence of the proposed window sidecars, `prepare-large` may use the
existing three files and their spike-index/time distribution as available
evidence, with reduced guarantees about raw-I/O locality and survey windows.
A dataset with neither raw traces nor pre-extracted waveforms must not claim
that Waveform/Feature View evidence is available.

The full export's identifier/fingerprint should combine an immutable sorting
asset ID when available, full spike count, and a digest of a canonical
row-ordered immutable spike array (`spike_times.npy` is one candidate). The
upstream reservoir must be verified against that *exact* source. A mutable
`spike_clusters.npy` hash must not be the sole identity used across curation
iterations. Byte-level hashes in examples are illustrative; the definitive
AIND provenance method is a pending joint decision.

Coordinate identity and assignment revision are separate: the latter binds the
initial full cluster assignments and metadata for replay. Each curated output
records its parent revision and journal digest. Validate relevant immutable
feature, acquisition and time-base provenance as well as spike rows; equal
timestamp arrays alone do not establish source identity. Production hashing
must stream file contents rather than read a complete array into memory.

## 4. Bounded-package Phy-side output

`prepare-large` creates a **new directory**, not an in-place rewrite of the
original. At minimum it contains ordinary, internally consistent Phy files
(`params.py`, `spike_times.npy`, `spike_templates.npy`, `spike_clusters.npy`,
channel/template arrays, and present per-spike attributes), plus:

| File | Role |
| --- | --- |
| `_phy_spike_global_ids.npy` | Required `int64[N_bounded]` sorted mapping `b -> g`, no duplicates |
| `_phy_curation_manifest.json` | Versioned local contract; immutable source ID/fingerprint, complete/bounded counts, durations, preparation algorithm/parameters/seed, evidence limitations and file names |
| `_phy_spikes_subset.{spikes,channels,waveforms}.npy` | Existing Phy-format stored waveforms, *remapped to bounded row indices* (if available) |
| Full-duration metrics | Separate time-binned evidence, described in Section 6; do not pretend proxy metrics are full metrics |

Unselected spike rows are absent from the GUI arrays, **not deleted from the
complete sorting**. Absolute `spike_times.npy` values remain in acquisition
sample coordinates and require a dtype capable of representing 1000 h at the
source sample rate (typically `int64`, not `int32`). For 30 kHz, 1000 h is
108 billion samples. Keep recording start/end, session/block boundaries, valid
acquisition exposure, and gaps distinct from periods of zero firing.

Do not rely on phylib's fallback of estimating `model.duration` from the last
spike when raw traces are absent. The future GUI metrics/time-axis integration
must take the authoritative duration/coverage metadata instead. Standard
`params.py` still identifies the bounded Phy dataset; the manifest augments it.

## 5. Preparation algorithm: scientific and memory constraints

A first version should:

1. Scan the **full** local sorting in a bounded-memory batch pass; calculate
   full-duration per-unit time bins and coverage/exposure.
2. Partition *valid acquisition time* into coarse strata. Select short shared
   contiguous survey windows at deterministic seeded offsets across sessions/
   blocks (illustrative defaults: one 30 s window per 1 h stratum).
3. Include **all units and all spikes within each chosen window** as initial
   bounded candidates; preserve source indices and absolute times.
4. Enforce configurable **total spike / RAM budgets**. Add shared rescue windows
   for underrepresented units, prioritizing benefit to several units and
   temporally separated evidence; retain bounded deterministic selection.
5. Where a waveform reservoir exists, favor windows with usable waveform
   coverage and report per-unit counts and uncovered regions. Selecting a
   window does not guarantee all its spikes have stored waveforms.
6. Write an atomic, versioned bounded package. Verify array-length alignment,
   sorted global IDs, waveform/feature alignment, provenance, coverage and
   source immutability before marking it usable.

These choices are **heuristics**, not a claim of unbiased sampling, complete
physiological representativeness, or equivalent CCG/ISI statistics. Never label
metrics from the bounded subset as full-data measurements. Windows identified
manually in a future Unit Stability View may motivate a regenerated package;
automated anomaly detection is out of scope.

The upstream `20_000` waveform cap per unit is **not** a mandatory local budget:
for 1000 units x 20k waveforms x 90 samples x 32 channels x 4 bytes it would
approach **230 GB** uncompressed. Extraction, transfer, storage and bounded
GUI consumption require independently configurable *total* byte/spike budgets.

## 6. Unit Stability evidence from the full recording

A Unit Stability View should work on ordinary 1-10 h recordings too. For a
selected cluster it plots time-binned **full-sorting** spike count / firing rate,
amplitude median/quantiles, depth/position and presence/coverage when those
quantities exist in the source. Use an explicit time axis, valid acquisition
exposure per bin, data-availability flags, and recording gaps. Counts divided
by wall-clock bin width are wrong for acquisition gaps. No missing metric should
be silently synthesized.

Initial storage may use regular homogeneous `.npy` arrays plus a small manifest,
with explicit row-to-cluster-ID mapping and bin edges; exact metric schema and
plot integration belong to the Unit Stability PR. At each full-data curation
iteration, merged/split clusters must receive **correctly recomputed or
incrementally updated** summaries, not merely the stale parent-unit curves.

Current `BaseController.get_mean_firing_rate()` divides the loaded cluster's
`n_spikes` by `model.duration`; this would severely undercount a bounded unit.
When a bounded package is detected, its full-count/exposure-aware value must
replace that calculation wherever presented. The same provenance distinction
applies to the Cluster View, correlograms, ISI and other proxy-based statistics.

Window statistics need explicit observation intervals. Do not connect ISIs or
CCG pairs across independent segments or unsupported gaps. CCG normalization
must use a declared lag-exposure estimator for those intervals; initially hide
unsupported normalized diagnostics. Whole-window sampling, rescue selection
and incomplete waveform coverage do not establish unbiased full-data quality
metrics or chronic biological unit identity.

## 7. Semantic curation and full-data replay

The current `Clustering` operates on arrays of loaded spike rows, creates new
cluster IDs during `assign/split`, and saves those local assignments. They
cannot by themselves define the fate of billions of unobserved spikes.

Before mutations in a bounded GUI session, record **versioned semantic events**
with preconditions, affected source clusters, deterministic operation parameters,
resolved cluster IDs and provenance. Track edits through existing undo/redo and
save. Apply them in order against the *complete* local sorting to write a new
curated output (never overwrite the original).

| Operation | Initial replay policy |
| --- | --- |
| Merge complete units | Reapply to every source spike in the units; replayable |
| Cluster quality/metadata labels | Replay per resulting cluster identity; replayable |
| Threshold split | Replay only if the predicate/feature definition is serialized and an equivalent full-data per-spike attribute exists; otherwise disallow |
| Lasso or arbitrary selected spike-ID split | Not replayable for unseen spikes by default; **block in bounded mode** unless a valid full-data rule is supplied |
| Undo/redo | Journal must represent the committed final event sequence and allow exact save/reopen, not duplicate reverted edits |

Unbounded/ordinary Phy behavior must remain unchanged. It is not acceptable to
silently treat a selection of bounded spike IDs as a valid full-unit split.
Existing `curation.json` merge proposals and `curation_review.json` are separate
producer/review artifacts; the new semantic event journal must not overwrite
either file. Re-running `prepare-large` after `apply-curation` may reuse the same
source spike/waveform bytes because curation changes assignments, not spike
identity.

For the first editable release, support merges and labels only. Keep all splits
blocked until their specific replay rules pass end-to-end tests. Allocate IDs
from the complete unit catalog, including units absent from the subset. Record
resolved inherited metadata, rather than recomputing it from full parent counts.
Commit journal, assignments and metadata as one validated save generation;
reopen must recover a coherent generation and allocator state. Validate operation
intent before mutation, including plugin and console entry points.

## 8. Test/reference fixture and future acceptance

`phy/apps/tests/large_recording_fixtures.py` deterministically writes a minimal
32-spike, **1000 h** full Phy-format sorting and a bounded eight-spike package,
including upstream waveforms for eight different source rows and remapped
bounded waveforms for five. The fixture intentionally omits the raw binary.
`phy/apps/tests/test_large_recording_contract.py` asserts:

- exact `int64` sample indexing well past the 32-bit limit, source-order and
  time preservation;
- full -> bounded -> waveform mapping, including copied waveform/channel rows;
- existing Phy waveform filenames, row/dtype/shape and `-1` padding invariants;
- simple window array dtypes, sample-coordinate bounds and shared row alignment;
- source fingerprint agreement, separate source-vs-bounded identity;
- empty/disjoint intersections and existing phylib load compatibility.

**These are reference contract tests only:** they do not claim that
`prepare-large`, Unit Stability or semantic replay already exists. Subsequent
PRs should reuse the fixture but replace the reference remapping with tests of
the production command. Add mismatched-source, reordered-row, excessive-memory,
partially covered units, gap, sparse feature and replay/undo corruption tests
before enabling bounded curation on real datasets.

## 9. Ownership and implementation order

1. **This PR:** Phy-side design, source-coordinate/format reference fixture and
   tests only. No GUI changes, no new CLI, no upstream exporter changes.
2. **Unit Stability:** full-data summaries and selected-unit GUI view, initially
   usable on normal recordings. Keep user-visible full/partial provenance clear.
3. **Semantic curation journal:** integration with `Supervisor`, event semantics,
   undo/save/reopen and blocking unsafe splits in bounded mode.
4. **`prepare-large`:** separate streaming batch CLI, input validation, window
   selection, output/remapped waveform files and bounded-manifest generation.
5. **`apply-curation`:** validate/replay semantic actions against full local
   sorting, write a new curated full sorting, verify, then regenerate a bounded
   package. Test deterministic repeatability and failures atomically.

The integration plan now organizes these stages as commits within a single PR,
rather than separate PRs. **Do not ship an editable bounded GUI without replay safety**.

Agent starting points: `AGENTS.md`, `design/README.md`,
`phy/apps/__init__.py` (Click CLI), `phy/apps/template/gui.py` (controller),
`phy/apps/base.py` (views/rates/save), `phy/cluster/{clustering,supervisor}.py`
(actions/history), `phy/apps/template/tests/test_gui.py`, and
`phy/apps/tests/test_large_recording_contract.py`. The existing waveform
reader is in the separate `phylib/io/model.py` dependency.

## 10. Decisions not frozen by this Phy-side PR

The upstream team should review the exact optional window-sidecar names, AIND
sorting asset identity/fingerprinting, segment/time/recording-block semantics,
waveform extraction alignment, total storage budget, and whether windows or a
more general sampling-plan format fits its pipeline best. Phy should accept the
existing three-file waveform format regardless. Values such as `1 h`, `30 s`,
`2,000` and `20,000` are preliminary, not schema requirements.

Code Ocean handles **creation of a reusable evidence reservoir**; Phy handles
**bounded selection, GUI coordinate remapping, full-duration summaries,
semantic curation and replay**. Neither side should infer an absent waveform
or silently redefine source spike IDs.

## Implementation references

- [Phy `Clustering.assign`](../phy/cluster/clustering.py) and
  [Supervisor](../phy/cluster/supervisor.py) for current row-based edits.
- [Template GUI](../phy/apps/template/gui.py) and
  [Waveform/Feature mixins](../phy/apps/base.py) for standard stored-waveform
  support and full/proxy metric safeguards.
- [Existing stored-waveform GUI test](../phy/apps/template/tests/test_gui.py).
- [SpikeInterface `export_to_phy` source (AIND inspected commit)](https://github.com/SpikeInterface/spikeinterface/blob/896ea840da72b4a5f06dcf8870ad820d20fe7031/src/spikeinterface/exporters/to_phy.py).
