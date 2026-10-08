# Repository guidance for agents

## Issue triage

- Before triaging or working on GitHub issues, read
  `.github/issue-audit/2026-07.yaml`.
- The audit is a point-in-time maintainer snapshot. GitHub is authoritative for
  current issue state, labels, comments, and newly reported regressions.
- Never close an issue because of age alone. For `status: needs-info` issues,
  check for a current reproduction before acting; the audit records the planned
  follow-up date and the information requested.
- Treat curation integrity, saved cluster assignments, split/merge behavior,
  channel mapping, and cross-view consistency as safety-sensitive. Require
  concrete evidence and regression coverage before declaring these fixed.
- Use the audit's `related_issues` and `retained_backlog.issue_families` fields
  to avoid creating duplicate investigations.

## Development workflow

- Use `uv` for development environments: `uv sync --dev`.
- For tests requiring unreleased phylib fixes, use the CI-matching environment
  recipe in `docs/installation.md#reproduce-the-ci-test-environment`.
- Run the narrowest relevant tests while iterating, for example
  `uv run pytest path/to/test_file.py`. Use `uv run --no-sync` when preserving
  a phylib source or CI-revision override.
- Before handing off a code change, run `make lint`, `make format-check`, and
  the relevant test target. Use `make test-full` for broad GUI, application, or
  data-model changes. Checks included by the chosen target satisfy these
  requirements; do not run them separately again unless a subsequent change
  affects their results.
- Run `make doc-check` for published documentation, public API, shortcut, or
  plugin documentation changes. It regenerates checked-in documentation and
  verifies that generated output is current. Standalone `AGENTS.md` and `design/`
  edits do not require it unless they also affect published documentation.
- Run `uv build` when changing packaging, dependencies, entry points, or package
  data.

## Change discipline

- Add focused regression tests for bug fixes, especially around selection
  updates, split/merge/undo, saving and backups, waveform/channel mapping, and
  cleanup of GUI or OpenGL resources.
- Consider large-dataset memory and rendering costs when changing views. Avoid
  unbounded allocations or work proportional to every spike during interactive
  updates.
- Add user-visible changes to the unreleased section of
  `docs/changelog.md`; pure tests, refactors, and formatting changes do not need
  entries.
- Keep installation guidance aligned with the `uv`-first workflow in
  `README.md` and `docs/installation.md`.
