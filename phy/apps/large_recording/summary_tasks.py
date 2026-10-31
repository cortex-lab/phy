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
    optimized: bool = False
    range_supported: bool = True
    additional_bytes: int = 0
    peak_bytes: int = 0
    budget_bytes: int = 0


@dataclass(frozen=True)
class StoreArtifact:
    path: str
    warning: str = ''
    reference: bytes | None = None
    amplitude_index: str = ''


def _check_binding(request, context):
    context.check_cancel()
    if request.dataset_id:
        manifest = json.loads((Path(request.package) / '_phy_curation_manifest.json').read_text())
        if manifest.get('preparation_identity', '') != request.dataset_id:
            raise ValueError('The prepared dataset changed; reopen it before starting this task.')


def inspect_summary_store(request, context):
    _check_binding(request, context)
    estimate = estimate_summary_store(request.package, source_dir=request.source)
    from .range_build import estimate_range_index
    from .range_index import AmplitudeRangeIndex
    from .range_intervals import RangeUnavailableError

    available = Path(estimate.source_path).is_dir()
    compatible, optimized, supported, path = False, False, True, ''
    with LiveSummaryEngine(request.package) as engine:
        try:
            with SummaryStore(engine) as store:
                compatible, path = True, str(store.path)
            available = True  # Existing rows are sufficient; original source may be offline.
        except (OSError, ValueError, KeyError, TypeError):
            pass
        try:
            ranges = estimate_range_index(engine)
            additional = ranges.final_bytes + (0 if compatible else estimate.estimated_bytes)
            peak = ranges.peak_extra_bytes + (0 if compatible else estimate.estimated_bytes)
            budget = ranges.final_bytes + estimate.estimated_bytes
        except RangeUnavailableError:
            supported = False
            additional = peak = budget = estimate.estimated_bytes
        if compatible and supported:
            try:
                index = AmplitudeRangeIndex(engine)
                try:
                    optimized = index.indexed_units == set(engine.manifest['unit_ids'])
                finally:
                    index.close()
            except (OSError, ValueError, KeyError, TypeError):
                pass
    return StoreInspection(
        estimate, compatible, available, path, optimized, supported, additional, peak, budget
    )


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

    _, reference = attach_summary_store(
        request.package,
        request.destination,
        expected_reference=request.expected_reference,
        before_attach=commit,
        check_cancel=context.check_cancel,
        return_reference=True,
    )
    candidate = Path(request.destination) / 'amplitude-index'
    return StoreArtifact(
        request.destination,
        reference=reference,
        amplitude_index=str(candidate) if candidate.is_dir() else '',
    )


def prepare_amplitude_index(request, context):
    from .range_build import build_amplitude_index
    from .summary_store import summary_store_reference

    _check_binding(request, context)
    if summary_store_reference(request.package, allow_invalid=True) != request.expected_reference:
        raise ValueError('The attached summary store changed; prepare its accelerator again.')

    def cancel():
        context.check_cancel()
        return False

    def commit():
        _check_binding(request, context)
        context.commit()

    def progress(phase, done, total):
        context.report(
            TaskProgress(
                phase.capitalize(),
                int(done),
                int(total),
                'rows' if phase == 'sorting amplitude ranges' else 'bytes',
            )
        )

    try:
        path = build_amplitude_index(
            request.package,
            request.destination,
            max_bytes=request.max_bytes,
            cancel=cancel,
            progress=progress,
            before_publish=commit,
            attach=False,
        )
        return StoreArtifact(str(path))
    except SummaryStorePublishedError as error:
        return StoreArtifact(str(error.path), str(error))


def connect_amplitude_index(request, context):
    from .range_index import attach_amplitude_index

    context.report(TaskProgress('Attaching amplitude index'))
    _check_binding(request, context)

    def commit():
        _check_binding(request, context)
        context.commit()

    attach_amplitude_index(
        request.package,
        request.destination,
        expected_reference=request.expected_reference,
        before_attach=commit,
        check_cancel=context.check_cancel,
    )
    return StoreArtifact(request.destination)
