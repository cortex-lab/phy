# phy 2.2 merge-workflow integration handoff

Status snapshot: 2026-08-08

This document records the remaining integration work around phy PR #1404. It
is a point-in-time handoff, not a substitute for GitHub. Before acting, read
`AGENTS.md` and `.github/issue-audit/2026-07.yaml`, then verify every PR state,
review, comment, head SHA, and check result on GitHub.

## 1. Branch and PR topology

Keep the existing two implementation branches. Do not create replacement or
stacked branches for this work.

- phy: `feature/merge-view-workflow`
  - PR: https://github.com/cortex-lab/phy/pull/1404
  - snapshot head: `d7ff0c5ee0762a04cea0b3451bfb4a3f7466a339`
  - snapshot state: draft, cleanly mergeable, all 12 CI checks passing
- phylib: `agent/fix-template-less-curation-reload`
  - PR: https://github.com/cortex-lab/phylib/pull/63
  - snapshot head: `fc494f6ab9f03370c43e618d2ef9610c6781b0e6`
  - snapshot state: draft and cleanly mergeable; phylib has no GitHub Actions
    workflow, so this PR has no automated checks

PR #1404 deliberately pins its GitHub Actions environment to phylib commit
`fc494f6ab9f03370c43e618d2ef9610c6781b0e6`. Keep that exact pin until a phylib
release contains the fix.

## 2. Current acceptance gate for PR #1404

PR #1404 is being manually retested by `@goatsofnaxos`. The current build fixes:

- template-less saved waveform subsets failing after save/reopen;
- repeated waveform warnings and Probe View falling back to channel zero;
- manual merges remaining in Merge mode instead of returning the merged unit
  as the sole Cluster View selection.

The retest request is:
https://github.com/cortex-lab/phy/pull/1404#issuecomment-5227384869

At this snapshot, that request is the latest comment and has no tester reply.
While waiting, keep `feature/merge-view-workflow` frozen at the tested head. Do
not merge `master` into it, rewrite it, or force-push it. Silence is not release
acceptance. If no response arrives after several business days, post one concise
follow-up rather than assuming success.

## 3. Correct merge order before final integration

Review and merge these PRs one at a time in this order:

1. phylib #63 — template-less stored waveform sample count
   - https://github.com/cortex-lab/phylib/pull/63
   - Run or verify `phylib/io/tests/test_model.py` (41 tests at the snapshot).
   - This repository has no CI, so human review and local evidence are the gate.
2. phy #1408 — correct `pc_features.npy` documentation
   - https://github.com/cortex-lab/phy/pull/1408
   - Documentation-only, with strict documentation CI passing.
3. phy #1406 — numeric sorting for text-valued table columns
   - https://github.com/cortex-lab/phy/pull/1406
   - Focused widget regressions and the full platform matrix pass.
4. phy #1407 — rescale Waveform View when waveform type changes
   - https://github.com/cortex-lab/phy/pull/1407
   - This belongs to the audit's waveform-correctness family. Preserve its
     focused regression covering every waveform-type switching entry point.
5. phy #1409 — grouped dependency lock refresh
   - https://github.com/cortex-lab/phy/pull/1409
   - Merge this last so `uv.lock` represents the final dependency state.

At the snapshot, phy #1406 through #1409 are non-draft, cleanly mergeable, have
the full green CI matrix, and have no reviews or comments. Recheck that this is
still true before approving or merging. Do not merge merely because CI is green;
review the current patch first.

The smaller phy PRs overlap #1404 as follows:

- #1406: `phy/gui/widgets.py`, its tests, and `docs/changelog.md`;
- #1407: `phy/cluster/views/waveform.py`, its tests, and the changelog;
- #1408: the changelog;
- #1409: `uv.lock`.

Merging them to `master` while #1404 is frozen is expected. Integrate `master`
into #1404 only once, after the manual-feedback gate below is satisfied.

## 4. Decision after tester feedback

If the tester confirms the fixes:

1. Record or acknowledge the acceptance on PR #1404.
2. Confirm phylib #63 and phy #1406 through #1409 are merged.
3. Merge current `origin/master` once into `feature/merge-view-workflow`.
4. Resolve overlap, validate, push, and wait for the complete PR CI matrix.
5. Request final code review and mark #1404 ready only when all gates pass.

If the tester reports another problem:

1. Do not integrate `master` yet; keep debugging the exact tested revisions.
2. Reproduce the report and add focused regression coverage, especially for
   waveform/channel mapping, saved curation, merge/undo, or cross-view state.
3. Put phylib changes only on the existing phylib branch and phy changes only
   on the existing phy branch.
4. Validate and push phylib first, update phy's exact phylib CI pin if its SHA
   changes, then push phy and request another retest.
5. Integrate `master` only after the tester accepts the corrected build.

## 5. Integrating master into PR #1404

Use a normal merge, not a rebase or force-push. The branch is shared for manual
testing, and preserving its published history keeps the tested revisions
traceable.

Before merging, ensure both the local worktree and remote tracking state are
clean. Then fetch and merge current `origin/master` into
`feature/merge-view-workflow`. Resolve conflicts deliberately:

- retain #1406's numeric-text sorting and focused widget tests;
- retain #1407's waveform-type bounds invalidation and regression test;
- retain #1408's corrected PC-feature documentation;
- combine all unreleased changelog entries without duplicating them;
- regenerate `uv.lock` from the resolved `pyproject.toml` with `uv`, then inspect
  the dependency diff rather than choosing either conflict side wholesale;
- retain the exact phylib CI pin until a phylib release supersedes it.

Do not use destructive history commands such as `git reset --hard`, and do not
discard unrelated or user-authored worktree changes.

## 6. Required final validation

After resolving the integration, follow `AGENTS.md` and run at minimum:

```bash
make lint
make format-check
make doc-check
make test-full
uv build
```

Also rerun the focused template-less waveform regression and the selection and
Supervisor suites against the tested phylib revision. Confirm that the full
GitHub Actions matrix passes on Linux, macOS, and Windows for Python 3.10–3.12,
plus docs, spelling, and build jobs. If CI fails, inspect Actions logs, implement
the narrowest fix, validate locally, commit and push, and repeat until green.

Manual release acceptance must cover a copy of a real dataset:

- save, close, and reopen a template-less curated dataset;
- select multiple units and confirm Waveform and Probe views use distinct,
  cluster-specific channels without repeated warnings;
- complete a manual merge and confirm Normal mode returns with only the result
  selected, quality assignment works immediately, and explicit re-entry permits
  another merge;
- undo and redo the merge and verify the exact workspace transitions;
- verify proposition merges still advance automatically;
- save and reopen once more to check curation integrity.

## 7. Ready-to-merge and post-merge gates

PR #1404 may be marked ready only when all of the following are true:

- explicit manual tester acceptance is recorded;
- phylib #63 is merged;
- phy #1406 through #1409 are merged and integrated;
- local required validation and the complete PR CI matrix pass;
- the real-dataset smoke test passes;
- a final reviewer has reviewed the integrated patch.

Prefer squash-merging PR #1404 because its branch contains a long development
history. After merge, verify `master` CI. When a phylib release containing
`fc494f6` (or its successor) is available, replace the temporary commit pin with
the released dependency, regenerate the lockfile, run packaging and CI checks,
and publish that cleanup separately.
