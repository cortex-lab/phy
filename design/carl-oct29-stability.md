# Carl's 29 October stability report

The console records checkout `6855aae4`, successful row-store and amplitude-index
preparation, rapid Similarity navigation errors, then a shutdown abort. The
accompanying 28 October PDF describes the older full-source scan cost. Preparation
success alone does not establish the path or latency of later GUI summary jobs.

## Selection and curation targets

Table selection changes immediately, while its events are debounced. Keeping
only the last incremental mutation lost transitions required by the canonical
selection reducer. Its strict `before_ids` guard correctly detected divergence.
Repeated Next actions could keep failing until the primary selection reset.

An isolated 12-spike bounded fixture also reproduced a target-safety issue:
a pending Next displayed primary/candidate `[0, 2]` while canonical state still
held `[0, 1]`. Immediate merge recorded parents `[0, 1]`; label and Move Similar
likewise affected the old candidate. This reproduction does not establish that
Carl's saved dataset was affected.

The fix retains a bounded ordered mutation batch on its own debounce key. A
scratch controller validates and reduces all intents before applying the result
atomically; expensive presentation is published once. Projections and resets
invalidate superseded generations. Overflow drains operations rather than
dropping them. Curation boundaries settle both tables before target resolution,
semantic preflight, transaction snapshots and action history capture. Reentrant
or nonsettling selection observers fail before a curation mutation.

Permanent regressions include rapid actual Next actions, mixed intent/color
semantics, hidden selections, reference resets, unrelated debounced events,
atomic failure, bounded draining and role transfers. Bounded direct/shortcut
merge, label and move tests check exact native targets, journal entries, history,
saved metadata and full-source replay. Graphical splitting rejects a preview
made obsolete by a pending reference change.

## Generic task shutdown

`TaskWidget.dispose()` evaluated a deleted manager's Qt signals before its
exception guard. Owner cleanup could then stop before cancellation. Independently,
a worker finishing after manager deletion could release its task key but strand
manager ownership because terminal delivery required the deleted receiver.

Generic workers now retain only a plain Python record/context. A GUI timer polls
bounded progress and terminal mailboxes. Detached managers continue terminal
polling without owner callbacks; QObject destruction cancels work and releases
manager ownership on the GUI thread. Worker completion never accesses Qt. Short
commit-aware publication and synchronous task chaining are preserved.

Widget and owner cleanup tolerate destruction order, including prompt preflight
tasks. Subprocess regressions cover deleted managers before active/committed
workers finish and full GUI destruction after idle/completed manager deletion.
The live-summary reader has a separate existing delivery mechanism; this change
does not replace every worker lifecycle in Phy.

## Runtime evidence and validation

The GUI integration regression covers source, row-store and amplitude-index
paths separately. The amplitude case spies successful range resolution for new
split expressions and rejects both row-store and source fallback. Existing
count/trend checks, save/reopen and full-source replay remain in that test. The
GUI preparation/handover test also asserts actual range queries after attachment.

One debug line per live-summary job reports path, unit count and elapsed time.
It distinguishes `amplitude-index`, `row-store`, `source` and `prepared/cache`;
there is no per-spike logging or new index algorithm in this fix.
