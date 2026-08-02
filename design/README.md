# Design documents

These documents describe proposed or in-progress changes that are not yet part
of the released user documentation.

## Merge View

Read these documents in order:

1. [Merge View workflow specification](merge-view-workflow.md) fixes the agreed
   user-visible behavior.
2. [Merge View architecture proposal](merge-view-architecture.md) describes the
   internal refactor and implementation path needed to support that behavior.

The workflow specification is the authority for user behavior. The architecture
proposal may evolve as implementation reveals constraints, but changes must not
silently alter the workflow contract.

### Current status

- The manual Merge View workflow and its supporting architecture are implemented
  on the feature branch.
- Merge Propositions and `curation.json` are intentionally deferred.
- The remaining work is release review and validation of the implemented
  workflow; Merge Propositions remain a separate future project.

Agents continuing this work should first read the repository `AGENTS.md`, then
both Merge View documents completely. Merge, selection, undo/redo, saved cluster
assignments, colors, and cross-view consistency are safety-sensitive; do not
declare the feature complete without the regression coverage and verification
listed in the architecture proposal.

## Amplitude-threshold splitting

The [amplitude-threshold splitting implementation plan](amplitude-threshold-splitting.md)
defines the user interaction, safety invariants, controller/view boundaries,
delegable work packages, and verification required for amplitude-based split
previews in Amplitude View and Waveform View.
