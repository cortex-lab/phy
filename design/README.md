# Design documents

These documents record implemented and proposed work targeting phy 2.2.0. User
documentation remains authoritative for released behavior.

## Merge View

Read these documents in order:

1. [Merge View workflow specification](merge-view-workflow.md) fixes the agreed
   user-visible behavior.
2. [Merge View architecture record](merge-view-architecture.md) describes the
   internal refactor supporting that behavior.
3. [Merge Propositions specification](merge-propositions.md) defines review of
   AIND/SpikeInterface format-version 2 `curation.json` merge propositions.

The workflow specification is the authority for user behavior. The architecture
record may evolve as implementation reveals constraints, but changes must not
silently alter the workflow contract.

### Current status

- Manual Merge View is implemented on the unreleased phy 2.2 branch.
- Merge Propositions are implemented for Template GUI datasets on that branch.
- Automated release validation is complete (`make test-full`, lint, formatting,
  strict documentation build, and package build). Remaining work is maintainer
  acceptance and manual dataset smoke testing before release.

Agents continuing this work should first read the repository `AGENTS.md`, then
both Merge View documents completely. Merge, selection, undo/redo, saved cluster
assignments, colors, and cross-view consistency are safety-sensitive; do not
declare the feature complete without the regression coverage and verification
listed in the architecture record and proposition specification.

## Amplitude-threshold splitting

The [amplitude-threshold splitting implementation plan](amplitude-threshold-splitting.md)
defines the user interaction, safety invariants, controller/view boundaries,
delegable work packages, and verification required for amplitude-based split
previews in Amplitude View and Waveform View.

The implementation and user documentation are complete on the unreleased phy
2.2 branch. Final large-dataset smoke testing and save/reopen validation remain.
