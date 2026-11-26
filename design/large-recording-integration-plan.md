# Long recording audit and integration plan

## Assessment

The downloaded contract bundle is a useful foundation. Keep the bounded GUI,
immutable full sorting, explicit source-row mapping, and full-data replay model.
Do not enable editing until replay and persistence agree with the GUI. A 1000-hour
timestamp fixture validates coordinate range; it does not validate capacity or
scientific representativeness.

This review targets phy `7e439491` and the downloaded
`phy_large_recording_contract_bundle.zip`. The accompanying contract remains a
proposal. This plan refines its implementation order and supplies additional
requirements; the phases below govern enablement.

## Findings that affect correctness

1. **Identity needs two layers.** A spike-coordinate fingerprint identifies
   immutable rows and their time/channel/feature provenance. An assignment
   revision identifies the starting full cluster assignments and metadata.
   Replay must validate both. A times-only hash cannot distinguish recordings
   with identical timestamps or changes to feature values. Include the acquisition
   identity, time base, relevant immutable array digests and schema in coordinate
   identity; hash assignments and metadata separately for each revision. Record
   parent revision, journal digest, output revision and preparation identity.
   Compute hashes incrementally, without `read_bytes()` on production arrays.

2. **Cluster IDs must come from the full namespace.**
   [Clustering](../phy/cluster/clustering.py) initializes its allocator from the
   maximum loaded cluster ID. A subset can omit the highest full ID. Seed the
   allocator from a full unit catalog and persist its high-water mark. Record
   explicit output IDs in events and reject collisions during replay. Units with
   no bounded evidence remain in the catalog and output, with missing evidence
   visible to the user.

3. **Save is a transaction.**
   [BaseController](../phy/apps/base.py) currently saves assignments and metadata
   separately; [Supervisor.save](../phy/cluster/supervisor.py) emits further save
   events and clears dirty state. Adding another journal file to that chain does
   not make it crash consistent. Bounded mode needs immutable save generations
   containing assignments, metadata, journal and identity, published by a single
   commit pointer. Reopen only complete validated generations. Preserve the last
   valid generation after interruption; keep dirty state after a failed commit.
   Define whether undo survives reopen. Recommended v1: reopen establishes a new
   undo baseline while retaining the committed semantic history for replay.

4. **Record resolved metadata.** `Clustering._do_merge` selects the largest
   parent using loaded counts, and Supervisor inherits its metadata. Full and
   sampled counts can choose different parents. Serialize the resolved group and
   custom labels in merge events, including an explicit policy for conflicting
   labels. Replay must not infer this choice from full counts.

5. **Capture intent before mutation.**
   [AmplitudeView.on_request_split](../phy/cluster/views/amplitude.py) returns
   selected row IDs and clears the threshold. A listener after clustering cannot
   recover the predicate. Route bounded edits through a command service before
   any mutation. Gate direct assign/split, plugins, console operations, shortcuts,
   merge proposals and threshold providers at the mutation boundary. V1 supports
   whole-unit merges and labels; block every split until its full-data semantics
   are implemented and tested.

6. **Statistics require observation intervals.** Preserve shared contiguous
   windows and segment boundaries. ISIs must not connect windows or acquisition
   gaps. CCGs must exclude pairs crossing unsupported boundaries, and their
   normalization must match observed lag exposure; the existing full-duration
   baseline in `BaseController._get_correlograms_rate` is not valid for a window
   subset. Initially suppress unsupported normalization and diagnostic claims.
   Label window statistics as sampled evidence. Rescue selection and waveform
   availability introduce bias; they do not justify extrapolating quality,
   contamination, refractory violations or chronic unit identity to the full run.

7. **Define time and channel coordinates.** V1 accepts one validated monotonic
   sample axis with a fixed sampling rate and explicit valid acquisition
   intervals. Reject unsupported segment layouts rather than concatenate silently.
   Preserve tied timestamps and source row order. Sparse waveform channel IDs
   index model channels; `channel_map.npy` maps these to raw-file columns. They
   are not interchangeable with hardware or SpikeInterface channel IDs. Test a
   nonidentity map, sparse padding, whitening/scaling and waveform alignment.
   Keep sample coordinates as integers and CPU times as float64; subtract the
   viewport origin before converting fine-scale plotting positions to float32.

8. **All preparation products need budgets.** A dense units-by-time-bin summary,
   exact quantile calculation, waveform gather, or full index inversion can
   exceed RAM despite streamed input. Specify batch, output, summary, cache and
   GUI memory limits, plus temporary disk estimates. Distinguish peak working
   memory from file-backed pages. Whole shared windows exceeding the spike budget
   must be shortened, rejected or replaced with documented deterministic rules;
   do not silently thin spikes within a window and retain its statistical claims.

9. **Persist the final window plan.** Store retained intervals, segment/block
   identity, survey/rescue classification and valid exposure in the bounded
   package. A seed and parameters do not identify trimmed or replaced windows.
   Count acquisition exposure once when composing merged-unit summaries.

## Module boundaries

Use a small `phy/apps/large_recording/` package, with pure services callable
without Qt and thin Click/controller adapters. Refine filenames while coding;
keep these responsibilities separate:

| Responsibility | Contract |
| --- | --- |
| Source adapter and validation | Stream canonical arrays; classify per-spike, template, channel and cluster data explicitly; reject ambiguous unsupported arrays |
| Manifest and identity | Validate versions, paths, coordinate identity, assignment revisions, exposure and capabilities |
| Preparation | Deterministic window plan, bounded output, streamed waveform remapping, budget report |
| Summaries | Full counts/exposure, availability, declared exact or approximate estimators and multiresolution queries |
| Semantic commands and journal | Operation validation, explicit IDs/metadata, history cursor and transactional commit |
| Replay | Pure event interpretation, chunked output assignments, invariants and lineage |
| GUI adapter | Capability gating, evidence labels, bounded model/controller integration |

Do not load the complete source through TemplateModel or Clustering in batch
commands. Do not hide full-data scans in a selected-unit GUI callback. Limit
changes to phylib to demonstrated loader/coordinate fixes, with separate tests
and dependency requirements. Preserve existing ordinary-dataset behavior.

## One implementation PR with ordered commits and acceptance gates

Implement the complete workflow on `feature/long-recording-curation` in the
main phy checkout. Use the stages below as reviewable commit groups within one
PR. Intermediate capability gates support development; merge the PR only after
the complete prepare, curate, save, replay and regenerate workflow passes its
acceptance checks. Predicate splits require their own full-data equivalence
checks within the same PR if included in its supported functionality.

### 1. Contract and reference fixtures

Start with the revised contract, this plan and format fixtures.
Fix the bundle's missing `dtype` in `params.py`, allow equal sample times, and
reject malformed reference row IDs. Run the real released-phylib loader test
without accepting a skip. Its success proves this small file layout loads,
not that all GUI workflows or long-recording statistics are supported.

### 2. Source validation and summary engine

Implement identities, assignment revisions, array inventory, exposure and
capability validation independently of Qt. Start with full counts and rates;
add amplitude/depth only with defined units and available source attributes.
Do not add per-bin quantiles until their exact/approximate algorithm, merge
behavior and memory costs are specified. Medians cannot be merged by averaging
parent medians; invalidate or recompute unavailable derived summaries.

Gate: exact tiny-oracle counts; missing exposure; gaps; source/feature mismatch;
reordered rows; column-vector input arrays; unknown versions; unsupported
segments; bounded peak memory as source size grows; actionable disk estimates.

### 3. Read-only preparation and inspection

Implement `prepare-large` with deterministic shared windows, rescue policies,
per-unit evidence reports and explicit budgets. Write into a temporary sibling
directory and publish only after complete validation. Rebuild paths in params,
subset every registered per-spike array, retain template-indexed structures, and
avoid copying stale `.phy` caches and derived cluster metrics. Unknown attributes
must be explicitly included, excluded with a report, or rejected.

Add a bounded-manifest GUI adapter with authoritative duration, full/proxy
counts and unsupported-view safeguards. Until phase 6, require read-only bounded
sessions, including save and plugin mutation entry points.

Gate: production command vs independently specified expected arrays; shuffled
channel mapping; actual `get_waveforms()` values; sparse PC feature alignment;
missing/partial waveform coverage; tied samples; zero evidence units; empty
selection policy; deterministic budgets; interrupted publication; no source
writes; GUI open/close and time-axis precision tests.

### 4. Semantic commands and durable saves

Implement merge/label command validation, full-catalog ID allocation, resolved
metadata, undo/redo active history and transactional generations. Keep the
feature disabled for editable bounded sessions while replay is incomplete.
Existing `curation.json` and `curation_review.json` retain their separate roles.

Gate: merge chains; conflicting labels; undo then branch; redo; reopen baseline;
omitted high-ID units; proposal-driven merges; rejected direct split/assign;
failed-save dirty state; fault injection at every commit boundary; assignments
equal the journal's projection onto bounded rows.

### 5. Full-data replay

Implement `apply-curation` against an exact source assignment revision. For v1
merges, resolve the final source-unit mapping once, then stream assignments into
a new full dataset. Preserve row count/order, times and immutable arrays; write
metadata and lineage coherently. Reject repeated application to the wrong parent
revision. Publish only a verified complete output. Reprepare from that revision
with the same coordinate identity and fresh revision-dependent summaries.

Gate: compare every assignment with an independent tiny oracle; preserve all
unseen spikes; replay chains and labels exactly; reject mismatched parent or
corrupted journal; interruption never exposes a complete-looking partial output;
same source/events produce the same semantic result; source bytes unchanged.

### 6. Enable bounded merges and labels with Unit Stability

Expose editable bounded mode only when phases 3–5 pass together. Integrate a
Unit Stability view that distinguishes acquisition exposure, evidence windows,
available measurements and unresolved identity. Add bounded status and operation
capabilities to the existing workflow. Count/exposure summaries can update for
merges; quantiles and other nonadditive metrics require recomputation or explicit
unavailability. Keep all splits blocked.

Gate: prepare → inspect → merge/label → undo/redo → save/reopen → replay →
reprepare → verify. Include ordinary 1–10 h datasets to prevent regressions.
Run `make test-full`, `make doc-check`, and `uv build` when CLI/package changes
are included. Complete manual smoke testing with an actual chronic dataset.

### 7. Optional predicate splits

Add one operation at a time after the initial workflow is accepted. Store feature
identity/units, predicate operator, threshold, missing-value policy, parent
revision and explicit child IDs. The same predicate must evaluate identically
on GUI evidence and all source spikes. Stored waveform-derived or local PCA
amplitudes do not automatically define a full-data predicate. Lasso splits
remain unavailable without a separately validated full-data rule.

## Standard recording compatibility gate

Recording length must never select a new behavior implicitly. Activate bounded
curation only through an explicit validated bounded manifest. Without that
manifest, retain the standard loader, selector, metrics, mutation capabilities,
save format and history behavior. A malformed bounded manifest must produce an
actionable error rather than silently enter standard editable mode. Optional
Unit Stability on ordinary recordings must not replace their existing views,
change their default layout or introduce full-source scans at GUI startup.

Keep the semantic command, replay and transactional save implementation behind
the bounded adapter wherever possible. Any necessary shared-code change requires
focused standard-mode regression coverage in addition to bounded-mode tests.
Do not impose bounded split restrictions or journal files on ordinary datasets.

Before merging the single PR, run `make test-full` and verify ordinary datasets
with raw traces, stored waveforms only, and sparse features. Cover standard
merge and label inheritance, lasso/threshold/template splits, undo/redo and
branching, save/reopen assignments and metadata, merge proposals, plugin-driven
edits, channel mapping, sampled views and existing rate/ISI/CCG behavior. Assert
that ordinary saves do not create bounded journals or manifests. Compare startup
memory and selection latency on the same ordinary datasets before and after the
change; any meaningful regression must be explained and resolved before merge.
Existing tests and these checks establish compatibility evidence; the plan
itself does not guarantee compatibility before implementation.

## Capacity and scientific acceptance

Use three kinds of evidence: tiny exact-oracle datasets for correctness;
generated high-density on-disk datasets for resource behavior; real chronic
recordings for usability and scientific acceptance. Duration, spike count,
number of units, array widths and available waveform coverage are separate axes.
Measure preparation/replay throughput, peak process memory, temporary disk,
GUI startup and selected-unit update latency on documented hardware. Establish
acceptance budgets from those measurements before advertising capacity.

Have a domain curator review known drift, rare/bursting units, transient
artifacts, gaps and intentionally incorrect merges. Compare selected evidence
with withheld contiguous windows and full-data diagnostics where available.
Record what the curator can and cannot infer; finite selected evidence cannot
guarantee biological identity throughout 1000 hours. Software integrity and
scientific evidence sufficiency are separate release criteria.

## Producer decisions

Agree with the producer team on source asset identity, full unit IDs, authoritative
duration/exposure, segment mapping, waveform time/channel/amplitude semantics,
reservoir coverage and total storage costs. Existing three-file waveforms remain
supported; proposed window sidecars must remain optional until jointly agreed.
The inspected [SpikeInterface exporter revision](https://github.com/SpikeInterface/spikeinterface/blob/896ea840da72b4a5f06dcf8870ad820d20fe7031/src/spikeinterface/exporters/to_phy.py)
supports one segment and distinguishes Phy channel indices from exported original
channel IDs. Do not assume that arbitrary multi-session inputs already satisfy
this contract or that a deployed exporter matches this inspected revision.

## Implementation and validation status

The branch now includes batched source validation and preparation, full-source
summaries, bounded GUI curation and checkpoints, semantic replay, Unit Stability,
and dedicated long-recording tests. The feature is implemented, but its
acceptance remains incomplete: this test and synthetic benchmark evidence does
not replace real chronic-recording review by domain curators.

On 2026-10-13, `make test-full` completed with 745 passed and 1 skipped. Three
additional bounded GUI lifecycle regressions pass in
`phy/apps/large_recording/tests/test_gui_lifecycle.py`.
[`design/large-recording-validation.md`](large-recording-validation.md) records
the synthetic capacity runs and their limits. The separate 32-spike, 1000-hour
fixture verifies sample-coordinate handling and file mapping; it is not a
billion-spike capacity test or evidence of chronic biological identity. Curator
acceptance on real chronic recordings remains outstanding.
