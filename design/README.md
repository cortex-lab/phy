# Design documents

These documents record implemented and proposed work targeting phy 2.2.0. User
documentation remains authoritative for released behavior.

## Merge View

Choose the documents and sections relevant to the change:

1. [Merge View workflow specification](merge-view-workflow.md) fixes the agreed
   user-visible behavior.
2. [Merge View architecture record](merge-view-architecture.md) describes the
   internal refactor supporting that behavior.
3. [Merge Propositions specification](merge-propositions.md) defines review of
   AIND/SpikeInterface format-version 2 `curation.json` merge propositions.
4. [Merge View dock stability plan](merge-view-dock-stability.md) records the
   persistent dock and atomic workspace switching to avoid layout disruption.
5. [Merge workflow feedback plan](merge-workflow-feedback-plan.md) records the
   September 2026 assessment and implemented transfer, reference-promotion, lookup,
   gesture, and column-layout work, with regression requirements for maintenance.

The workflow specification is the authority for user behavior. The architecture
record may evolve as implementation reveals constraints, but changes must not
silently alter the workflow contract.

### Current status

- Manual Merge View is implemented on the unreleased phy 2.2 branch.
- Merge Propositions are implemented for Template GUI datasets on that branch.
- Automated release validation is complete (`make test-full`, lint, formatting,
  strict documentation build, and package build). Remaining work is maintainer
  acceptance and manual dataset smoke testing before release.
- The dated [integration handoff](merge-view-integration-handoff.md) records the
  PR dependencies, manual-feedback gate, merge order, conflict policy, and final
  validation steps at that snapshot. GitHub remains authoritative for live PR state.

Follow the repository `AGENTS.md`. Use the workflow specification for behavior
changes, architecture sections 3–8 for state, events, history, and view integration,
and section 10 for regression and performance coverage. Use the proposition
specification for proposition review and persistence. A substantial redesign may
require reviewing the full workflow and architecture together.

Merge, selection, undo/redo, saved cluster assignments, colors, and cross-view
consistency are safety-sensitive; do not declare the feature complete without
the regression coverage and verification listed in the architecture record and
proposition specification. Historical implementation sequences and delegation
assignments describe the original work, not requirements to restart it.

## Amplitude-threshold splitting

The [amplitude-threshold splitting implementation plan](amplitude-threshold-splitting.md)
records the user interaction, safety invariants, controller/view boundaries,
original work packages, and verification required for amplitude-based split
previews in Amplitude View and Waveform View. Use sections 2–4 for behavior and
state contracts, the relevant package's acceptance tests for focused changes,
and section 11 for integration and manual acceptance.

The implementation and user documentation are complete on the unreleased phy
2.2 branch. Final large-dataset smoke testing and save/reopen validation remain.

## Long-recording curation (proposed)

The [long-recording curation Phy-side design contract](large-recording-curation.md)
describes bounded local datasets for ~1000-hour sorting datasets, the three spike-index
spaces, optional upstream waveform evidence, full-duration metrics, replay safety,
and the implementation stages. The accompanying synthetic 1000-hour
fixture and reference tests are in `phy/apps/tests/large_recording_fixtures.py`
and `phy/apps/tests/test_large_recording_contract.py`.

This is a proposal and a reference test contract, **not yet** a `prepare-large`
implementation or an upstream export-format agreement.

The [integration audit and plan](large-recording-integration-plan.md) refines
the contract with correctness findings, module boundaries, ordered commits in one PR and
acceptance gates. Follow its gates before enabling editable bounded sessions.
