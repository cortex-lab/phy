# Merge Propositions workflow and implementation specification

Status: implemented and automatically validated on the unreleased phy 2.2 branch;
manual dataset smoke testing and release acceptance remain

Companion documents:

- [Merge View workflow specification](merge-view-workflow.md)
- [Merge View architecture record](merge-view-architecture.md)

## 1. Goal

Allow a curator to review automatic merge suggestions from an AIND/SpikeInterface
format-version 2 `curation.json` file through the existing Merge View. Reviewing,
editing, or cancelling a proposition must not bypass the ordinary merge path.
Only `G` commits cluster assignments.

This is a curation-integrity feature. Proposition input, review decisions,
clustering assignments, undo/redo context, and saved output must remain mutually
consistent.

## 2. Input contract

Phy reads dataset-local `curation.json` as an AIND/SpikeInterface curation model.
The first implementation consumes these fields:

```json
{
  "format_version": "2",
  "unit_ids": [41, 56, 72],
  "merges": [
    {"unit_ids": [41, 56]},
    {"unit_ids": [56, 72], "new_unit_id": 1000}
  ]
}
```

Other top-level fields, including `label_definitions`, `manual_labels`,
`removed`, and `splits`, are preserved as input but are not applied by the Merge
Propositions workflow in phy 2.2.

For each `merges` entry:

- `unit_ids` is an ordered list containing at least two unique integer cluster
  IDs;
- the first ID is the blue reference;
- every ID must appear in the file's top-level `unit_ids` and in the clustering
  loaded by phy before the proposition can be reviewed;
- `new_unit_id` is accepted and preserved as provenance, but phy continues to
  allocate merge result IDs through its ordinary clustering model; and
- an internal stable key is derived from the ordered `unit_ids`. Exact duplicate
  entries are invalid rather than silently coalesced.

Unsupported format versions or invalid top-level JSON disable the proposition
workflow with a clear warning but never prevent ordinary curation. An invalid
individual merge entry remains visible with its reason when the rest of the
file can be decoded safely.

## 3. Review state and persistence

`curation.json` is producer-owned input and is never overwritten by phy.
Dataset-local `curation_review.json` stores phy review decisions:

```json
{
  "format_version": "1",
  "source": {
    "filename": "curation.json",
    "sha256": "..."
  },
  "reviews": {
    "merge:0123456789abcdef": {
      "decision": "accepted",
      "applied_unit_ids": [41, 56],
      "result_unit_id": 1001
    }
  }
}
```

Persisted decisions are `accepted` and `rejected`. `pending` is represented by
the absence of a decision. `accepted_modified` is derived when an accepted
record's `applied_unit_ids` differ from its source proposition. `invalid` and
`stale` are derived from the source and current clustering and are never stored
as curator decisions.

The source hash detects replacement of `curation.json`. Matching proposition
keys retain their decisions after a source update; unmatched old review records
are preserved as orphaned provenance and reported rather than applied to another
proposition.

Review state participates in phy's dirty/save lifecycle. On Save, phy first
saves cluster assignments and metadata through the existing model path, then
atomically writes `curation_review.json` through a temporary sibling and replace.
The dirty state is cleared only after both succeed. Closing with unsaved review
decisions uses the existing save prompt.

## 4. User-visible workflow

When a valid `curation.json` contains merges, phy creates a persistent **Merge
Propositions** view. It shows status, proposition key, ordered cluster IDs,
cluster count, reference, and `new_unit_id` when present.

Available operations are:

- **Review**: open the proposition in Merge mode;
- **Reject**: record a reversible rejection without changing clustering;
- **Skip / Next pending**: navigate without changing review state; and
- **Reset review**: return an accepted or rejected proposition to pending when
  its source clusters still exist.

Invalid and stale propositions remain visible with a reason but cannot be
reviewed.

Review is explicit; ordinary row selection does not enter Merge mode. Starting a
review snapshots the complete current Normal workspace, then stages the ordered
proposition IDs directly. It must not first project those IDs into Cluster View,
because cancellation must restore the curator's pre-review state.

While reviewing:

- the existing Merge View, Similarity View, transfer, reorder, and color rules
  remain authoritative;
- the Merge status identifies the source proposition;
- the curator may add, remove, or reorder candidates;
- `V`, Cancel, or closing Merge View restores the exact Normal entry snapshot
  and leaves the proposition pending; and
- another proposition cannot be reviewed concurrently.

On `G`, phy calls the ordinary merge implementation. Only after that call
succeeds does it record the proposition as accepted, including the actual
ordered merge IDs and result cluster ID. A changed workspace produces the
derived `accepted_modified` status. Failure leaves the workspace and review
state unchanged.

Reject creates a review-history entry and is undoable. Skip does not create
history. Reset review is explicit and undoable.

## 5. Overlap, stale IDs, and clustering changes

Overlapping propositions are allowed in the input. After any merge or split,
pending propositions are revalidated against live cluster IDs. A proposition
with a missing source ID becomes stale and cannot be partially applied.

Phy never remaps a stale proposition through clustering descendants. In
particular, a split may produce several descendants, and choosing one
automatically could commit an unintended merge.

Undoing an accepted proposition restores:

- the original spike assignments;
- the proposition's prior review state;
- the complete pre-commit Merge workspace;
- the Normal-entry snapshot used by cancellation; and
- derived validity of overlapping propositions.

Redo reapplies the merge, restores the accepted decision, and exits Merge mode.

## 6. Architecture boundaries

Durable proposition state is separate from `CurationSelectionState`:

```text
MergePropositionController
        | begin review
        v
CurationSelectionController
        | successful G
        v
Clustering + proposition resolution in one GlobalHistory entry
```

The pure proposition module owns decoding, validation, review transitions,
serialization, and live-ID validity projection. It performs no Qt, filesystem,
or spike-array work.

`MergeSession` gains optional proposition provenance. The selection controller
gains one atomic operation that stages arbitrary validated proposition IDs while
capturing the current Normal snapshot.

`Supervisor` remains the workflow facade. The application/controller layer owns
dataset paths and file I/O. Neither the proposition view nor the selection state
loads or writes JSON.

The initial implementation applies to Template GUI datasets. Legacy Kwik
support is deferred.

## 7. Required regression coverage

- Format-v2 round trip, optional fields, unsupported versions, malformed JSON,
  duplicate entries, missing IDs, and mixed valid/invalid merges.
- Stable proposition keys and source-file change detection.
- Exact Normal-state cancellation after starting a review.
- Ordered staging with the first ID fixed as blue reference.
- Invalid and stale propositions cannot enter Merge mode.
- Failed merge preserves review state and the complete workspace.
- Successful and modified acceptance record exact applied and result IDs.
- Overlapping propositions become stale without automatic remapping.
- Manual merge and split safely revalidate pending propositions.
- Undo/redo couples assignments, review decisions, colors, presentation, table
  context, and Merge workspace.
- Reject/reset undo and redo.
- Save/reopen after accept and reject, atomic-write failure, and external source
  replacement.
- View close/reopen and controller shutdown release callbacks.
- Catalog updates operate on cluster/proposition IDs and never scan spikes.

## 8. Implementation packages

1. Pure proposition domain, codec, and tests.
2. Standalone Merge Propositions view and Qt tests.
3. Atomic review persistence adapter and tests.
4. Selection, Supervisor, history, and failure-atomicity integration.
5. BaseController loading/saving and application regressions.
6. User documentation, changelog, generated documentation, and final audit.

Packages 1-3 may run in parallel after this contract is frozen. Package 4 has a
single integration owner because selection, merge, undo/redo, and saved curation
are safety-sensitive.
