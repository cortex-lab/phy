"""Immutable adapters between summary snapshots and generic GUI tasks."""

import json
from dataclasses import dataclass
from pathlib import Path

from phy.gui.tasks import TaskProgress

from .live_summaries import LiveSummaryEngine
from .summary_store import (
    ReferenceStamp,
    StoreEstimate,
    SummaryStore,
    SummaryStorePublishedError,
    attach_summary_store,
    build_summary_store,
    estimate_summary_store,
)


@dataclass(frozen=True)
class StoreRequest:
    package: str
    source: str | None = None
    destination: str = ''
    expected_reference: bytes | ReferenceStamp | None = None
    max_bytes: int = 68719476736
    dataset_id: str = ''


@dataclass(frozen=True)
class StoreInspection:
    estimate: StoreEstimate
    compatible: bool
    source_available: bool
    path: str = ''


@dataclass(frozen=True)
class StoreArtifact:
    path: str
    warning: str = ''


def _check_binding(request, context):
    context.check_cancel()
    if request.dataset_id:
        manifest = json.loads((Path(request.package) / '_phy_curation_manifest.json').read_text())
        if manifest.get('preparation_identity', '') != request.dataset_id:
            raise ValueError('The prepared dataset changed; reopen it before starting this task.')


def inspect_summary_store(request, context):
    _check_binding(request, context)
    estimate = estimate_summary_store(request.package, source_dir=request.source)
    try:
        with LiveSummaryEngine(request.package) as engine, SummaryStore(engine) as store:
            return StoreInspection(
                estimate, True, Path(estimate.source_path).is_dir(), str(store.path)
            )
    except (OSError, ValueError, KeyError, TypeError):
        return StoreInspection(estimate, False, Path(estimate.source_path).is_dir())


def prepare_summary_store(request, context):
    _check_binding(request, context)

    def cancel():
        context.check_cancel()
        return False

    def commit():
        _check_binding(request, context)
        context.commit()

    def progress(phase, done, total):
        stages = {
            'validating source': ('Validating source', 'bytes'),
            'building summary store': ('Building summary store', 'rows'),
            'finalizing summary store': ('Finalizing', ''),
        }
        stage, unit = stages[phase]
        context.report(
            TaskProgress(stage, int(done) if total else None, int(total) if total else None, unit)
        )

    try:
        path = build_summary_store(
            request.package,
            request.destination,
            source_dir=request.source,
            max_bytes=request.max_bytes,
            cancel=cancel,
            progress=progress,
            attach=False,
            before_publish=commit,
        )
        return StoreArtifact(str(path))
    except SummaryStorePublishedError as error:
        return StoreArtifact(str(error.path), str(error))


def connect_summary_store(request, context):
    context.report(TaskProgress('Attaching summary store'))
    _check_binding(request, context)

    def commit():
        _check_binding(request, context)
        context.commit()

    attach_summary_store(
        request.package,
        request.destination,
        expected_reference=request.expected_reference,
        before_attach=commit,
        check_cancel=context.check_cancel,
    )
    return StoreArtifact(request.destination)
