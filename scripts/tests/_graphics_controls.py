"""Maintainer-only controlled-fault validation; not an ordinary user diagnostic."""

import importlib.util
import sys
from pathlib import Path

from OpenGL import GL  # Match Phy's import order before importing Qt.
from PyQt5.QtCore import QCoreApplication, QPoint, QPointF, Qt, QTimer
from PyQt5.QtGui import QWheelEvent

kit_path, control, input_mode, destination = sys.argv[1:]
Path(destination).mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location('kit', kit_path)
kit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kit)
original_watch, original_emit = kit.watch_surface, kit.emit
windows = {}
fired = False


def watch(canvas, label):
    windows[label] = canvas
    original_watch(canvas, label)


def emit(event, **fields):
    global fired
    original_emit(event, **fields)
    phase = 4 if control == 'hide' else 1
    target = 0 if control == 'hide' else 1
    if (
        not fired
        and event == 'update-request'
        and fields.get('phase') == phase
        and fields.get('canvas') == target
        and fields.get('role') == 'initial'
    ):
        fired = True

        def inject():
            canvas = windows[target]
            original_emit(
                'injected-control',
                kind=control,
                input_mode=input_mode,
                note='Deliberate maintainer validation, not a naturally occurring fault',
            )
            if control == 'hide':
                canvas.diagnostic_dock.hide()
            elif control == 'color':
                canvas.visual.set_data(pos=[[0.0, 0.0]], color=(0.0, 0.0, 1.0, 1.0), size=80.0)
            else:
                wheel = QWheelEvent(
                    QPointF(20, 20),
                    QPointF(20, 20),
                    QPoint(0, 0),
                    QPoint(0, 1200),
                    Qt.NoButton,
                    Qt.NoModifier,
                    Qt.ScrollUpdate,
                    False,
                )
                QCoreApplication.sendEvent(canvas, wheel)

        QTimer.singleShot(0, inject)


kit.watch_surface, kit.emit = watch, emit
raise SystemExit(
    kit.main(
        [
            '--_probe',
            'docked',
            '--input-mode',
            input_mode,
            '--_artifacts',
            destination,
            '--timeout',
            '20',
        ]
    )
)
