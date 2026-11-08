#!/usr/bin/env python3
"""Passive ordinary TemplateController probe; use only with disposable data.

Runs the application's normal Qt event loop and default dock layout. The only
intentional interactions are Supervisor.select, real Qt table mouse clicks,
and one injected wheel event. No observation requests a redraw or reads GL.
"""

import argparse
import hashlib
import importlib
import json
import logging
import os
import sys
import tempfile
import threading
import time
import traceback
from collections import Counter
from functools import wraps
from pathlib import Path


def _status(phases, errors, reason):
    checks = [
        [v for v in p.get('views', []) if v.get('check') in ('PASS', 'FAIL')] for p in phases
    ]
    if errors or any(v['check'] == 'FAIL' for phase in checks for v in phase):
        return 'FAIL'
    if len(phases) == 4 and all(checks) and reason == 'completed':
        return 'PASS'
    return 'INCOMPLETE'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path, help='Disposable dataset directory or params.py')
    parser.add_argument('--trace', type=Path, required=True, help='Output JSON trace and summary')
    parser.add_argument('--log', type=Path, required=True, help='Output Python/Qt debug log')
    parser.add_argument('--scratch', type=Path, help='Parent for a fresh isolated state directory')
    parser.add_argument(
        '--inject-worker-error',
        action='store_true',
        help='Maintainer control: deliberately fail a Qt worker; the result must be FAIL',
    )
    parser.add_argument(
        '--duration', type=float, default=26, help='Total seconds, between 20 and 28'
    )
    args = parser.parse_args()
    if not 20 <= args.duration <= 28:
        parser.error('--duration must be between 20 and 28 seconds')
    params = args.dataset.resolve()
    if params.is_dir():
        params /= 'params.py'
    if not params.is_file():
        parser.error('dataset params.py does not exist')
    for path in (args.trace, args.log):
        path.parent.mkdir(parents=True, exist_ok=True)
    scratch_parent = args.scratch or args.trace.resolve().parent
    scratch_parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix='ordinary-state-', dir=scratch_parent))
    started = time.monotonic()
    events, phases, errors = [], [], []
    lock = threading.RLock()
    canvases, counts, owners = {}, {}, {}
    state = {'phase': 'startup', 'controller': None, 'gui': None, 'finished': False}
    info = {
        'harness_revision': 3,
        'dataset': str(params),
        'scratch': str(scratch),
        'duration': args.duration,
        'environment': {
            key: os.environ.get(key)
            for key in (
                'QT_QPA_PLATFORM',
                'QT_XCB_GL_INTEGRATION',
                'PYOPENGL_PLATFORM',
                'XDG_SESSION_TYPE',
                'WAYLAND_DISPLAY',
                'DISPLAY',
                'QT_SCALE_FACTOR',
                'QT_LOGGING_RULES',
                'LIBGL_ALWAYS_SOFTWARE',
                'LP_NATIVE_VECTOR_WIDTH',
            )
        },
        'limitations': [
            'Paint/frameSwapped counts do not establish correct pixels or desktop presentation.',
            'Wheel and clicks are injected Qt events, not physical input or Wayland seat delivery.',
            'Fresh state/cache/config; default packaged layout; no user plugins or settings.',
            'Tracing and timers can change timing; no rendering observation performs GL readback.',
            'Controller cache/state/log paths redirected; no curation/save actions are performed.',
        ],
    }

    def small(value):
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, (tuple, list)):
            return [small(v) for v in value[:12]] if len(value) <= 12 else {'length': len(value)}
        if hasattr(value, 'shape'):
            return {'shape': list(value.shape), 'dtype': str(getattr(value, 'dtype', ''))}
        try:
            return value.item()
        except (AttributeError, ValueError):
            return type(value).__name__

    def emit(event, **fields):
        with lock:
            if len(events) < 100000:
                events.append(
                    {
                        't': round(time.monotonic() - started, 6),
                        'phase': state['phase'],
                        'event': event,
                        **fields,
                    }
                )
            else:
                info['trace_truncated'] = True

    def write_trace(reason):
        with lock:
            status = _status(phases, errors, reason)
            payload = {
                'status': status,
                'metadata': info,
                'reason': reason,
                'phases': list(phases),
                'errors': list(errors),
                'events': list(events),
            }
            state['status'] = status
            target = args.trace.with_suffix(args.trace.suffix + '.tmp')
            target.write_text(json.dumps(payload, indent=2, default=str) + '\n')
            target.replace(args.trace)

    def watchdog():
        emit('hard-timeout')
        write_trace('hard-timeout; Qt/native code did not exit within bound')
        os._exit(124)

    deadline = threading.Timer(args.duration + 2, watchdog)
    deadline.daemon = True
    deadline.start()
    handler = logging.FileHandler(args.log, mode='w')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(handler)

    class FailureHandler(logging.Handler):
        def emit(self, record):
            message = record.getMessage()
            if (
                record.levelno >= logging.ERROR
                or record.exc_info
                or 'Exception in paintGL' in message
                or 'Exception in initializetGL' in message
                or 'Exception in initializeGL' in message
            ):
                text = f'{record.levelname} {record.name}: {message}'
                if record.exc_info:
                    text += '\n' + ''.join(traceback.format_exception(*record.exc_info))
                with lock:
                    errors.append(text)
                emit('logged-error', message=text)

    root.addHandler(FailureHandler())
    write_trace('startup')

    try:
        # This import follows Phy's original ordering: OpenGL before PyQt.
        # Keep a statement boundary so isort cannot move Qt ahead of GL.
        importlib.import_module('OpenGL.GL')
        from OpenGL import platform
        from phylib.io.model import load_model
        from phylib.utils import connect
        from PyQt5.QtCore import (
            QCoreApplication,
            QEvent,
            QObject,
            QPoint,
            QPointF,
            Qt,
            QTimer,
            qInstallMessageHandler,
        )
        from PyQt5.QtGui import QOpenGLContext, QWheelEvent
        from PyQt5.QtTest import QTest

        import phy
        import phy.apps.base as base_module
        from phy.apps.template.gui import TemplateController
        from phy.cluster.views.base import ManualClusteringView
        from phy.gui.qt import Worker, create_app
        from phy.plot.base import BaseCanvas
        from phy.utils.context import Context

        info.update(
            phy_version=phy.__version__,
            phy_source=str(Path(phy.__file__).resolve()),
            pyopengl_platform=type(platform.PLATFORM).__name__,
            python=sys.version,
        )
        package = Path(phy.__file__).parent
        info['source_sha256'] = {
            name: hashlib.sha256((package / name).read_bytes()).hexdigest()
            for name in (
                '__init__.py',
                'plot/base.py',
                'gui/qt.py',
                'gui/widgets.py',
                'cluster/views/base.py',
                'cluster/supervisor.py',
            )
        }

        def qt_message(kind, context, message):
            emit('qt-message', kind=int(kind), category=context.category, message=message)
            level = logging.ERROR if int(kind) in (2, 3) else logging.WARNING
            logging.getLogger('qt').log(level, '%s: %s', context.category, message)
            if int(kind) == 3:  # QtFatalMsg precedes native abort.
                errors.append('Qt fatal: ' + message)
                write_trace('qt-fatal')

        qInstallMessageHandler(qt_message)

        def exception_hook(kind, value, tb):
            message = ''.join(traceback.format_exception(kind, value, tb))
            errors.append(message)
            emit('uncaught-exception', traceback=message)
            logging.getLogger('probe').error(message)

        sys.excepthook = exception_hook

        def worker_error(payload):
            message = payload[2] if len(payload) > 2 else repr(payload)
            with lock:
                errors.append('Worker error: ' + message)
            emit('worker-error', traceback=message)

        original_worker_init = Worker.__init__

        @wraps(original_worker_init)
        def worker_init(worker, *positional, **keywords):
            original_worker_init(worker, *positional, **keywords)
            worker.signals.error.connect(worker_error, Qt.DirectConnection)

        Worker.__init__ = worker_init

        def label(canvas):
            owner = owners.get(id(canvas))
            return (
                getattr(owner, 'name', type(owner).__name__) if owner else f'canvas-{id(canvas):x}'
            )

        def surface(canvas):
            try:
                owner = owners.get(id(canvas))
                context = canvas.context()
                panzoom = getattr(canvas, 'panzoom', None)
                return {
                    'view': label(canvas),
                    'selected': small(getattr(owner, 'cluster_ids', None)),
                    'visible': canvas.isVisible(),
                    'exposed': canvas.isExposed(),
                    'geometry': list(canvas.geometry().getRect()),
                    'dpr': canvas.devicePixelRatio(),
                    'context_valid': bool(context and context.isValid()),
                    'current_matches': bool(
                        context and QOpenGLContext.currentContext() == context
                    ),
                    'lazy': getattr(canvas, '_is_lazy', None),
                    'zoom': small(getattr(panzoom, 'zoom', None)),
                    'pan': small(getattr(panzoom, 'pan', None)),
                    'dock_visible': owner.dock.isVisible()
                    if owner and hasattr(owner, 'dock')
                    else None,
                }
            except RuntimeError:
                return {'view': label(canvas), 'deleted': True}

        event_names = {
            int(getattr(QEvent, name)): name
            for name in (
                'UpdateRequest',
                'Expose',
                'Resize',
                'Show',
                'Hide',
                'PlatformSurface',
                'Wheel',
                'MouseButtonPress',
                'MouseButtonRelease',
                'FocusIn',
                'FocusOut',
            )
        }

        class Observer(QObject):
            def eventFilter(self, obj, event):
                name = event_names.get(int(event.type()))
                if name:
                    emit('qt-event', qt_event=name, **surface(obj))
                return False

        app = create_app()
        info['qt_platform'] = app.platformName()
        observer = Observer(app)
        original_init = BaseCanvas.__init__

        @wraps(original_init)
        def canvas_init(canvas, *positional, **keywords):
            original_init(canvas, *positional, **keywords)
            key = id(canvas)
            canvases[key], counts[key] = canvas, Counter()
            canvas.installEventFilter(observer)

            def swapped():
                counts[key]['swaps'] += 1
                emit('frame-swapped', **surface(canvas))

            if hasattr(canvas, 'frameSwapped'):
                canvas.frameSwapped.connect(swapped)
            emit('canvas-created', **surface(canvas))

        BaseCanvas.__init__ = canvas_init
        for method, counter in (
            ('update', 'updates'),
            ('paintGL', 'paints'),
            ('initializeGL', 'initializations'),
        ):
            original = getattr(BaseCanvas, method)

            def wrap_canvas(original, method, counter):
                @wraps(original)
                def wrapped(canvas, *positional, **keywords):
                    counts.setdefault(id(canvas), Counter())[counter] += 1
                    emit(method + '-enter', **surface(canvas))
                    try:
                        return original(canvas, *positional, **keywords)
                    finally:
                        emit(method + '-exit', **surface(canvas))

                return wrapped

            setattr(BaseCanvas, method, wrap_canvas(original, method, counter))

        original_select = ManualClusteringView.on_select_threaded

        @wraps(original_select)
        def select(view, sender, cluster_ids, *positional, **keywords):
            emit(
                'view-select-enter',
                view=getattr(view, 'name', type(view).__name__),
                requested=small(cluster_ids),
                before=small(view.cluster_ids),
                lock=bool(view._lock),
                auto_update=view.auto_update,
            )
            try:
                return original_select(view, sender, cluster_ids, *positional, **keywords)
            except Exception:
                emit(
                    'view-select-error',
                    view=getattr(view, 'name', type(view).__name__),
                    traceback=traceback.format_exc(),
                )
                raise
            finally:
                emit(
                    'view-select-exit',
                    view=getattr(view, 'name', type(view).__name__),
                    selected=small(view.cluster_ids),
                    lock=bool(view._lock),
                )

        ManualClusteringView.on_select_threaded = select

        # Redirect bookkeeping only; do not clear or load any personal state.
        def isolated_cache(controller, clear_cache=None):
            controller.cache_dir = scratch / 'cache'
            controller.context = Context(controller.cache_dir)

        TemplateController._set_cache = isolated_cache
        base_module._add_log_file = lambda path: None  # --log already owns the log file.

        @connect(event='view_attached')
        def attached(view, gui):
            canvas = getattr(view, 'canvas', None)
            if isinstance(canvas, BaseCanvas):
                owners[id(canvas)] = view
                emit('view-attached', **surface(canvas))

        model = load_model(params)
        controller = TemplateController(
            model=model, dir_path=params.parent, config_dir=scratch / 'config', plugins=[]
        )
        state['controller'] = controller

        @connect(event='select', sender=controller.supervisor)
        def selected(sender, cluster_ids, **keywords):
            emit(
                'supervisor-select',
                requested=small(cluster_ids),
                kwargs={key: small(value) for key, value in keywords.items()},
            )

        write_trace('creating-ordinary-gui')
        gui = controller.create_gui(do_prompt_save=False)
        state['gui'] = gui

        def snapshot_counts():
            return {key: dict(value) for key, value in counts.items()}

        def begin_phase(name, action, expected=None):
            state['phase'] = name
            phase = {
                'name': name,
                'before': snapshot_counts(),
                'expected_clusters': expected,
                'started': round(time.monotonic() - started, 6),
            }
            phases.append(phase)
            emit('action-start', name=name)
            try:
                action()
            except Exception:
                message = traceback.format_exc()
                errors.append(message)
                emit('action-error', traceback=message)
            logging.getLogger('probe').info('Action: %s', name)

        def checkpoint():
            if not phases or 'views' in phases[-1]:
                return
            phase = phases[-1]
            phase['supervisor_selected'] = small(controller.supervisor.selected)
            phase['views'] = []
            for key, canvas in canvases.items():
                before = phase['before'].get(key, {})
                delta = {
                    name: counts[key][name] - before.get(name, 0)
                    for name in ('updates', 'paints', 'swaps')
                }
                observed = surface(canvas)
                eligible = (
                    observed.get('visible')
                    and observed.get('exposed')
                    and observed.get('dock_visible') is not False
                )
                if phase['name'] == 'wheel':
                    eligible = eligible and observed['view'] == state.get('wheel_target')
                    changed = state.get('wheel_before') != state.get('wheel_after')
                else:
                    changed = observed.get('selected') == phase['expected_clusters']
                check = (
                    ('PASS' if changed and delta['paints'] > 0 else 'FAIL') if eligible else 'SKIP'
                )
                phase['views'].append(
                    {
                        **observed,
                        **delta,
                        'check': check,
                        'missing_paint': bool(eligible and delta['paints'] == 0),
                    }
                )
            if phase['expected_clusters'] is not None:
                phase['selection_matches'] = (
                    phase['supervisor_selected'] == phase['expected_clusters']
                )
                if not phase['selection_matches']:
                    errors.append('Supervisor selection did not match ' + phase['name'])
            phase.pop('before', None)
            emit('checkpoint', name=phase['name'], views=phase['views'])
            write_trace('observing')

        def click(cluster_id):
            table = controller.supervisor.cluster_view
            index = table._proxy_index_for_id(cluster_id)
            if not index.isValid():
                raise RuntimeError('Requested cluster row is unavailable')
            widget = table.table_view
            widget.scrollTo(index)
            rect = widget.visualRect(index)
            if not rect.isValid():
                raise RuntimeError('Requested cluster has no visible Qt row rectangle')
            emit('table-click', cluster_id=cluster_id, rect=list(rect.getRect()))
            QTest.mouseClick(widget.viewport(), Qt.LeftButton, Qt.NoModifier, rect.center())

        def wheel():
            candidates = [
                canvas
                for key, canvas in canvases.items()
                if canvas.isVisible()
                and canvas.isExposed()
                and getattr(canvas, 'panzoom', None) is not None
            ]
            if not candidates:
                raise RuntimeError('No visible exposed panzoom canvas for wheel injection')
            canvas = next(
                (c for c in candidates if label(c).startswith('CorrelogramView')), candidates[0]
            )
            point = QPointF(canvas.width() / 2, canvas.height() / 2)
            event = QWheelEvent(
                point,
                QPointF(canvas.mapToGlobal(point.toPoint())),
                QPoint(0, 0),
                QPoint(0, 120),
                Qt.NoButton,
                Qt.NoModifier,
                Qt.ScrollUpdate,
                False,
            )
            emit('wheel-before', **surface(canvas))
            state['wheel_target'] = label(canvas)
            state['wheel_before'] = surface(canvas).get('zoom')
            QCoreApplication.sendEvent(canvas, event)
            state['wheel_after'] = surface(canvas).get('zoom')
            emit('wheel-after', accepted=event.isAccepted(), **surface(canvas))

        def finish():
            if state['finished']:
                return
            state['finished'] = True
            checkpoint()
            emit('observation-complete')
            state['phase'] = 'cleanup'
            try:
                gui.close()
                controller.close()
            except Exception:
                errors.append(traceback.format_exc())
            write_trace('completed' if len(phases) == 4 else 'incomplete-action-sequence')
            app.quit()

        ready_timer = QTimer(app)
        ready_timer.setInterval(200)

        def ready():
            if not controller.supervisor.cluster_view.is_ready():
                return
            ids = [int(value) for value in controller.supervisor.clustering.cluster_ids[:3]]
            if len(ids) < 2:
                errors.append('At least two clusters are required')
                finish()
                return
            ready_timer.stop()
            if args.inject_worker_error:

                def fail_worker():
                    raise RuntimeError('Deliberately injected worker failure')

                QTimer.singleShot(2400, lambda: Worker(fail_worker).run())
            info['cluster_ids'] = ids
            emit('application-ready', views=[surface(canvas) for canvas in canvases.values()])
            third = ids[-1] if len(ids) > 2 else ids[0]
            actions = [
                (
                    1200,
                    'supervisor-select',
                    lambda: controller.supervisor.select([ids[0]]),
                    [ids[0]],
                ),
                (5200, 'table-click-second', lambda: click(ids[1]), [ids[1]]),
                (9200, 'table-click-third', lambda: click(third), [third]),
                (13200, 'wheel', wheel, None),
            ]
            for offset, name, action, expected in actions:
                QTimer.singleShot(
                    offset,
                    lambda name=name, action=action, expected=expected: begin_phase(
                        name, action, expected
                    ),
                )
                QTimer.singleShot(offset + 3000, checkpoint)

        ready_timer.timeout.connect(ready)
        gui.show()
        ready_timer.start()
        remaining = max(1, int((args.duration - (time.monotonic() - started)) * 1000))
        QTimer.singleShot(remaining, finish)
        emit('event-loop-enter')
        app.exec_()
        if not state['finished']:
            write_trace('event-loop-ended-early')
        print(
            json.dumps(
                {
                    'trace': str(args.trace),
                    'log': str(args.log),
                    'scratch': str(scratch),
                    'status': state.get('status'),
                    'phases': len(phases),
                    'errors': len(errors),
                }
            )
        )
        return 0 if state.get('status') == 'PASS' else 1
    except Exception:
        message = traceback.format_exc()
        errors.append(message)
        emit('startup-error', traceback=message)
        logging.getLogger('probe').exception('Probe failed')
        write_trace('startup-error')
        print(message, file=sys.stderr)
        return 1
    finally:
        deadline.cancel()
        handler.flush()


if __name__ == '__main__':
    raise SystemExit(main())
