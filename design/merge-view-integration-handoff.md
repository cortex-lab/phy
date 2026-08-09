# phy 2.2 merge-workflow integration handoff

Status snapshot: 2026-08-09

This document records the remaining integration work around phy PR #1404. It
is a point-in-time handoff, not a substitute for GitHub. Before acting, read
`AGENTS.md` and `.github/issue-audit/2026-07.yaml`, then verify every PR state,
review, comment, head SHA, and check result on GitHub.

## 1. Branch and PR topology

Keep the existing two implementation branches. Do not create replacement or
stacked branches for this work.

- phy: `feature/merge-view-workflow`
  - PR: https://github.com/cortex-lab/phy/pull/1404
  - snapshot head before this handoff update:
    `0a47882d6cd6ca8190b77df29d489963da077cc3`
  - snapshot state: draft, manual feedback pending; all 12 checks passed on
    that head
- phylib: former branch `agent/fix-template-less-curation-reload`
  - PR: https://github.com/cortex-lab/phylib/pull/63
  - merged head: `fc494f6ab9f03370c43e618d2ef9610c6781b0e6`
  - merged into phylib `master` as
    `d9beeae0f8500f9e8a879003f5226a4b212a4b93`; the remote branch was deleted

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

### First executable batch

This batch is safe for a maintainer with merge and commit rights to execute
without waiting for independent review. Execute it serially so each later PR is
validated against the master branch produced by the previous step:

1. Run the complete local phylib test suite on phylib #63, fix any regression
   on its existing branch, wait for a clean result, and squash-merge #63.
2. From the updated phylib `master`, create `agent/add-ci`, add a focused GitHub
   Actions workflow for tests and package building, open a PR, and iterate on
   that branch until its own Actions checks pass. Then squash-merge the CI PR.
   Do not make legacy repository-wide lint failures a new blocking gate in this
   first workflow.
3. Recheck and squash-merge phy #1408.
4. Recheck phy #1406 against the new `master`; if GitHub reports a conflict or
   stale required checks, update its existing branch, preserve its focused
   regression, and wait for green CI before squash-merging it.
5. Apply the same gate to phy #1407 and squash-merge it last in this batch.

After every merge, verify the merged state on GitHub before starting the next
step. If a check fails, inspect its logs, make the narrowest fix on the PR's
existing branch, run the relevant local check, push, and repeat until green. Do
not update or merge `master` into phy #1404 during this batch: that branch stays
frozen pending the recorded manual tester response. When the batch ends, update
this handoff with the merged PR numbers, resulting master SHAs, checks run, and
the first unexecuted step.

### First batch result (completed 2026-08-09)

The batch above is complete:

- phylib #63 was squash-merged as
  `d9beeae0f8500f9e8a879003f5226a4b212a4b93` after all 273 local tests passed.
- phylib #64 added CI and repaired the source-distribution manifest. It was
  squash-merged as `d99464bddfcd9f0bdc1005dbe03f4510d67919cd` after its PR
  and post-merge `master` runs passed. Linux Python 3.10–3.12 and macOS Python
  3.12 run all 273 tests; Windows Python 3.12 runs the stable 60-test electrode,
  statistics, and utilities baseline; the build job verifies and uploads both
  source and wheel artifacts. The full Windows I/O suite remains excluded
  because it has pre-existing open-memory-map teardown failures.
- phy #1408 was squash-merged as
  `bcf01c10ec9309e241656028554962fa34d8a0e9`.
- phy #1406 was squash-merged as
  `1e56305eb50a5321079fb11740267c3cd29dfdfc`.
- phy #1407 was updated from its existing contributor branch after the earlier
  changelog merges, then squash-merged as
  `92f080080047af2013162027e253ba22cbfd22a8`. Its focused waveform tests, lint,
  formatting, strict documentation checks, and refreshed 12-check PR matrix
  all passed.

The first unexecuted merge step is phy #1409. Do not start the #1404 integration
until #1409 is merged and the manual tester feedback gate in section 2 is met.

### Full pre-integration order

Review and merge these PRs one at a time in this order:

1. phylib #63 — template-less stored waveform sample count
   - https://github.com/cortex-lab/phylib/pull/63
   - Run or verify `phylib/io/tests/test_model.py` (41 tests at the snapshot).
   - Completed as `d9beeae0f8500f9e8a879003f5226a4b212a4b93`.
2. phy #1408 — correct `pc_features.npy` documentation
   - https://github.com/cortex-lab/phy/pull/1408
   - Documentation-only, with strict documentation CI passing.
   - Completed as `bcf01c10ec9309e241656028554962fa34d8a0e9`.
3. phy #1406 — numeric sorting for text-valued table columns
   - https://github.com/cortex-lab/phy/pull/1406
   - Focused widget regressions and the full platform matrix pass.
   - Completed as `1e56305eb50a5321079fb11740267c3cd29dfdfc`.
4. phy #1407 — rescale Waveform View when waveform type changes
   - https://github.com/cortex-lab/phy/pull/1407
   - This belongs to the audit's waveform-correctness family. Preserve its
     focused regression covering every waveform-type switching entry point.
   - Completed as `92f080080047af2013162027e253ba22cbfd22a8`.
5. phy #1409 — grouped dependency lock refresh
   - https://github.com/cortex-lab/phy/pull/1409
   - Merge this last so `uv.lock` represents the final dependency state.

At this snapshot only #1409 remains in this sequence. Recheck its current patch,
head, mergeability, reviews, comments, and complete CI matrix before merging it.
Do not merge merely because CI is green.

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
