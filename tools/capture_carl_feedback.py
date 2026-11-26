"""Exercise Carl's feedback in the real GUI on a disposable data copy.

Starts its own Xvfb server: no window or recording reaches the desktop. Inputs
are hashed and copied before opening; curation is saved only in the copy.
Records real X11 pixels, Qt input events and state assertions with separate config.
"""

import argparse
import ctypes
import ctypes.util
import json
import os
import shutil
import subprocess
import time
from copy import deepcopy
from pathlib import Path


def run(args):
    if args.dataset.resolve() in (args.output.resolve(), *args.output.resolve().parents):
        raise ValueError('The capture output must be outside the immutable input dataset.')
    args.output.mkdir(parents=True, exist_ok=False)
    server_log = (args.output / 'xvfb.log').open('w')
    server = subprocess.Popen(
        [
            'Xvfb',
            '-displayfd',
            '1',
            '-screen',
            '0',
            '1920x1200x24',
            '-nolisten',
            'tcp',
            '-ac',
            '+extension',
            'GLX',
            '+render',
        ],
        stdout=subprocess.PIPE,
        stderr=server_log,
        text=True,
    )
    display = ':' + server.stdout.readline().strip()
    if display == ':':
        raise RuntimeError('Private X server failed to start.')
    os.environ.update(
        DISPLAY=display,
        QT_QPA_PLATFORM='xcb',
        LIBGL_ALWAYS_SOFTWARE='1',
        PYOPENGL_PLATFORM='glx',
        LP_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='2',
        OMP_NUM_THREADS='2',
    )
    recording = None
    controller = gui = app = None
    try:
        import numpy as np
        import OpenGL.GL  # Initialize OpenGL before Qt, as in the application entrypoint.
        from phylib.io.model import load_model
        from PyQt5.QtCore import QCoreApplication, QEvent, QPoint, QTimer
        from PyQt5.QtGui import (
            QColor,
            QCursor,
            QPainter,
            QPen,
            QPixmap,
            QPolygon,
        )
        from PyQt5.QtTest import QTest
        from PyQt5.QtWidgets import QInputDialog, QLineEdit, QMenu

        from phy.apps.large_recording.file_copy import copy_source_file
        from phy.apps.large_recording.source import hash_file
        from phy.apps.template.gui import TemplateController
        from phy.gui.qt import QApplication, Qt

        source_dataset = args.dataset.resolve()
        files = [
            p
            for p in source_dataset.iterdir()
            if p.is_file() and p.name not in ('phy.log', 'phy_curation_history.jsonl')
        ]
        source_hashes = {p.name: hash_file(p) for p in files}
        args.dataset = args.output / 'working-dataset'
        shutil.copytree(
            source_dataset,
            args.dataset,
            copy_function=copy_source_file,
            ignore=shutil.ignore_patterns('.phy', 'phy.log', 'phy_curation_history.jsonl'),
        )
        assert not any(p.is_symlink() for p in args.dataset.iterdir())
        renderer = subprocess.run(['glxinfo', '-B'], capture_output=True, text=True, check=True)
        (args.output / 'renderer.txt').write_text(renderer.stdout)
        app = QApplication.instance() or QApplication([])
        report = {
            'mode': args.mode,
            'dataset': str(source_dataset),
            'working_copy': str(args.dataset),
            'capture_scope': (
                'Settled X11 pixels; native XTest context click and Qt input events; '
                'not a spontaneous-redraw test'
            ),
            'private_display': display,
            'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'steps': [],
            'input_events': [],
            'recording_fps': args.fps,
            'pointer': 'Visible native X11 cursor, eased motion and click ring',
            'label': 'Real earlier Carl dataset; engineering validation',
        }
        bounded = args.mode == 'bounded'
        controller = TemplateController(
            model=None if bounded else load_model(args.dataset / 'params.py'),
            dir_path=args.dataset,
            config_dir=args.output / 'config',
            plugins=[],
            enable_threading=False,
        )
        bounded = args.mode == 'bounded'
        display_columns = ['id', 'group', 'n_spikes', 'ch']
        if bounded:
            display_columns.insert(2, 'full_count')
        controller.supervisor.columns = display_columns
        report['display_columns'] = display_columns
        names = [
            'SampledCorrelogramView' if bounded else 'CorrelogramView',
            'WaveformView',
            'EvidenceAmplitudeView' if bounded else 'AmplitudeView',
            'FeatureView',
        ]
        gui = controller.create_gui(
            default_views=names, view_count=dict.fromkeys(names, 1), do_prompt_save=False
        )
        if hasattr(controller, '_summary_store_actions'):
            controller._summary_store_actions._checked = True
        supervisor = controller.supervisor
        controller.n_spikes_waveforms = 50
        controller.n_spikes_waveforms_total = 350
        gui.resize(1920, 1200)
        gui.move(0, 0)
        gui.setWindowTitle('Carl feedback — ' + args.mode + ' real-data walkthrough')
        gui.show()

        def settle(ms=500):
            supervisor.block()
            app.processEvents()
            QTest.qWait(ms)
            app.processEvents()

        settle()
        science = [gui.get_view(name) for name in names]
        if any(view is None for view in science):
            raise RuntimeError(f'Missing scientific view: {names}')
        ccg, waveform, amplitude, feature = science
        waveform.waveforms_type = 'waveforms'
        if not bounded:
            amplitude.amplitudes_type = 'template'
        tables = [
            supervisor.cluster_view,
            supervisor.similarity_view,
            supervisor.merge_propositions_view,
        ]
        for view in tables:
            gui.addDockWidget(Qt.LeftDockWidgetArea, view.dock)
        for a, b in zip(tables[:-1], tables[1:]):
            gui.splitDockWidget(a.dock, b.dock, Qt.Vertical)
        gui.addDockWidget(Qt.RightDockWidgetArea, ccg.dock)
        gui.splitDockWidget(ccg.dock, waveform.dock, Qt.Horizontal)
        gui.splitDockWidget(ccg.dock, amplitude.dock, Qt.Vertical)
        gui.splitDockWidget(waveform.dock, feature.dock, Qt.Vertical)
        gui.resizeDocks([tables[0].dock, ccg.dock, waveform.dock], [370, 760, 760], Qt.Horizontal)
        gui.resizeDocks([v.dock for v in tables], [330, 300, 440], Qt.Vertical)
        settle()
        started = time.monotonic()
        video_log = (args.output / 'ffmpeg.log').open('w')
        recording = subprocess.Popen(
            [
                'ffmpeg',
                '-nostdin',
                '-y',
                '-video_size',
                '1920x1200',
                '-framerate',
                str(args.fps),
                '-f',
                'x11grab',
                '-draw_mouse',
                '1',
                '-i',
                display + '.0',
                '-an',
                '-c:v',
                'libx264',
                '-preset',
                'veryfast',
                '-crf',
                '20',
                '-pix_fmt',
                'yuv420p',
                '-movflags',
                '+faststart',
                str(args.output / 'walkthrough.mp4'),
            ],
            stdout=video_log,
            stderr=video_log,
        )

        def cursor_image(clicked=False):
            pixmap = QPixmap(72, 72)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            if clicked:
                painter.setPen(QPen(QColor('#ffcc4d'), 4))
                painter.drawEllipse(QPoint(28, 28), 22, 22)
            painter.setPen(QPen(QColor('#101010'), 2))
            painter.setBrush(QColor('white'))
            painter.drawPolygon(
                QPolygon(
                    [
                        QPoint(28, 28),
                        QPoint(28, 53),
                        QPoint(35, 47),
                        QPoint(41, 61),
                        QPoint(48, 58),
                        QPoint(42, 45),
                        QPoint(53, 45),
                    ]
                )
            )
            painter.end()
            return QCursor(pixmap, 28, 28)

        idle_cursor, clicked_cursor = cursor_image(), cursor_image(True)
        app.setOverrideCursor(idle_cursor)

        def input_note(label):
            entry = {'seconds': round(time.monotonic() - started, 3), 'label': label}
            report['input_events'].append(entry)
            print(json.dumps({'input': entry}), flush=True)
            return entry

        def show_cursor(cursor):
            # Application busy-cursor cleanup can remove an earlier override.
            # Re-establish our recording cursor without changing application code.
            if app.overrideCursor() is None:
                app.setOverrideCursor(cursor)
            else:
                app.changeOverrideCursor(cursor)

        def move_pointer(widget, pos):
            show_cursor(idle_cursor)
            origin = QCursor.pos()
            target = widget.mapToGlobal(pos)
            for step in range(1, 17):
                fraction = step / 16
                eased = fraction * fraction * (3 - 2 * fraction)
                QCursor.setPos(
                    round(origin.x() + (target.x() - origin.x()) * eased),
                    round(origin.y() + (target.y() - origin.y()) * eased),
                )
                QTest.qWait(25)
            QTest.qWait(120)

        def click(widget, button, modifiers=Qt.NoModifier, pos=None, label=None):
            pos = pos if pos is not None else widget.rect().center()
            button_name = 'Right click' if button == Qt.RightButton else 'Left click'
            prefix = 'Ctrl + ' if modifiers & Qt.ControlModifier else ''
            event = input_note(prefix + button_name + (': ' + label if label else ''))
            move_pointer(widget, pos)
            event['point'] = [QCursor.pos().x(), QCursor.pos().y()]
            show_cursor(clicked_cursor)
            QTest.qWait(100)
            event['click_seconds'] = round(time.monotonic() - started, 3)
            QTest.mousePress(widget, button, modifiers, pos)
            QTest.qWait(100)
            QTest.mouseRelease(widget, button, modifiers, pos)
            show_cursor(clicked_cursor)
            QTest.qWait(200)
            show_cursor(idle_cursor)

        def native_context_click():
            # Route the context-menu gesture through X11/XTest into Qt's xcb
            # platform plugin, rather than manufacturing a ContextMenu event.
            assert os.environ['DISPLAY'] == display
            x11 = ctypes.CDLL(ctypes.util.find_library('X11'))
            xtst = ctypes.CDLL(ctypes.util.find_library('Xtst'))
            x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
            x11.XOpenDisplay.restype = ctypes.c_void_p
            x11.XFlush.argtypes = [ctypes.c_void_p]
            x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
            xtst.XTestFakeButtonEvent.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint,
                ctypes.c_int,
                ctypes.c_ulong,
            ]
            xtst.XTestFakeButtonEvent.restype = ctypes.c_int
            connection = x11.XOpenDisplay(display.encode())
            if not connection:
                raise RuntimeError('Cannot open the private X11 display for context click.')
            try:
                for pressed in (1, 0):
                    if not xtst.XTestFakeButtonEvent(connection, 3, pressed, 0):
                        raise RuntimeError('XTest context-menu button event failed.')
                x11.XFlush(connection)
            finally:
                x11.XCloseDisplay(connection)

        def key(widget, value, modifiers=Qt.NoModifier, label='Keyboard shortcut'):
            event = input_note(label)
            QTest.qWait(250)
            event['key_seconds'] = round(time.monotonic() - started, 3)
            QTest.keyClick(widget, value, modifiers)
            event['key_released_seconds'] = round(time.monotonic() - started, 3)
            QTest.qWait(250)

        def capture(name, title, window=None, **facts):
            target = window or gui
            if window is None:
                gui.resize(1920, 1200)
                gui.move(0, 0)
            gui.statusBar().showMessage(title)
            settle(800)
            frame = target.frameGeometry()
            pixmap = (
                app.primaryScreen()
                .grabWindow(0)
                .copy(frame.x(), frame.y(), frame.width(), frame.height())
            )
            path = args.output / (name + '.png')
            assert pixmap.save(str(path), 'PNG')
            entry = {
                'name': name,
                'title': title,
                'seconds': round(time.monotonic() - started, 3),
                'selected': list(map(int, supervisor.selected)),
                **facts,
            }
            report['steps'].append(entry)
            (args.output / 'capture.json').write_text(json.dumps(report, indent=2))
            print(json.dumps(entry), flush=True)

        def click_row(table, unit, modifiers=Qt.NoModifier):
            index = table._proxy_index_for_id(unit)
            assert index.isValid(), unit
            table.table_view.scrollTo(index)
            settle(150)
            row = table._row_for_proxy_index(index)
            label = row.get('proposition') or f'Cluster {unit}'
            click(
                table.table_view.viewport(),
                Qt.LeftButton,
                modifiers,
                table.table_view.visualRect(index).center(),
                label=str(label),
            )
            settle()

        catalog = supervisor.merge_propositions.catalog
        choices = [
            p
            for p in catalog.propositions
            if len(p.unit_ids) == 2
            and all(unit in supervisor.clustering.cluster_ids for unit in p.unit_ids)
        ]
        assert choices, 'The real-data fixture needs a live pair SLAY suggestion.'
        proposal = next(
            (p for p in choices if bounded and tuple(p.unit_ids) == (25, 27)), choices[0]
        )
        first_parents = list(map(int, proposal.unit_ids))
        report['workflow'] = 'recursive-restore-and-exact-slay-reassembly'
        report['propositions'] = [{'key': proposal.key, 'unit_ids': first_parents}]
        report['first_parents'] = first_parents
        initial = supervisor.clustering.spike_clusters.copy()
        initial_events = list(controller.bounded_session.events) if bounded else None
        propositions = supervisor.merge_propositions_view
        if bounded:
            # Select real retained evidence for the initial pair. Later candidates
            # must also have stored waveforms in this same inspected window.
            stored = controller.model.spike_waveforms.spike_ids
            stored_labels = initial[stored]
            slots = np.searchsorted(
                controller.evidence_windows[:, 1],
                controller.model.spike_samples[stored],
                side='right',
            )
            counts = np.stack(
                [
                    np.bincount(
                        slots[stored_labels == unit], minlength=len(controller.evidence_windows)
                    )
                    for unit in first_parents
                ]
            )
            score = (counts > 0).sum(axis=0) * 10000 + counts.sum(axis=0)
            window_index = int(np.argmax(score))
            assert np.all(counts[:, window_index] > 0)
            controller.inspect_evidence_window(window_index)
            amplitude.amplitude_scope = 'overview'
            report['evidence_window'] = window_index
            report['window_waveforms_by_unit'] = dict(
                zip(map(str, first_parents), map(int, counts[:, window_index]))
            )
        for index, unit in enumerate(first_parents):
            click_row(
                supervisor.cluster_view, unit, Qt.ControlModifier if index else Qt.NoModifier
            )
        capture('01-initial', 'Compare the two source clusters before the first merge')

        def assert_slay_status(accepted):
            status = supervisor.merge_propositions.catalog.status_for(proposal.key).value
            row_id = propositions._id_by_key[proposal.key]
            row = propositions._model.row_by_id(row_id)
            assert row['_catalog_status'] == status, (row, status)
            color = propositions._foreground_color(row, 'proposition')
            if accepted:
                assert status == 'accepted', status
                assert color is not None and color.name() == '#86d16d', (row, color)
            else:
                assert status == 'accepted_restored', status
                assert color is None, (row, color)
            return status

        click_row(propositions, propositions._id_by_key[proposal.key])
        gui.activateWindow()
        app.setActiveWindow(gui)
        gui.setFocus()
        key(gui, Qt.Key_G, label='G: merge the first two source clusters')
        settle()
        first_merged = int(supervisor.selected[0])
        expected = initial.copy()
        expected[np.isin(initial, first_parents)] = first_merged
        np.testing.assert_array_equal(expected, supervisor.clustering.spike_clusters)
        accepted_review = dict(
            supervisor.merge_propositions.catalog.reviews[proposal.key].mapping()
        )
        assert accepted_review['result_unit_id'] == first_merged
        assert_slay_status(True)
        report['first_merged'] = first_merged
        capture('02-first-merge', 'First merge accepted: the SLAY suggestion is green')

        candidates = [
            int(unit)
            for unit in supervisor.similarity_view.get_ids()
            if int(unit) not in set(first_parents) | {first_merged}
            and (not bounded or np.any((stored_labels == unit) & (slots == window_index)))
        ]
        assert candidates, 'No similar third source cluster has evidence in the selected window.'
        third = candidates[0]
        click_row(supervisor.cluster_view, first_merged)
        click_row(supervisor.cluster_view, third, Qt.ControlModifier)
        key(gui, Qt.Key_G, label='G: merge the first result with one more source cluster')
        settle()
        merged = int(supervisor.selected[0])
        originals = first_parents + [third]
        expected[np.isin(initial, originals)] = merged
        np.testing.assert_array_equal(expected, supervisor.clustering.spike_clusters)
        report['nested_merge'] = {
            'immediate_inputs': [first_merged, third],
            'original_components': originals,
            'result': merged,
        }
        report['merged'] = merged
        report['union'] = originals
        report['nested_merge_slay_status'] = supervisor.merge_propositions.catalog.status_for(
            proposal.key
        ).value
        capture(
            '03-nested-merge',
            'Second merge contains the first merge plus a third source cluster',
            immediate_parents=[first_merged, third],
            original_components=originals,
        )

        unrelated_inputs = [
            int(unit)
            for unit in supervisor.clustering.cluster_ids
            if int(unit) not in set(originals) | {first_merged, merged}
        ][:2]
        assert len(unrelated_inputs) == 2
        click_row(supervisor.cluster_view, unrelated_inputs[0])
        click_row(supervisor.cluster_view, unrelated_inputs[1], Qt.ControlModifier)
        key(gui, Qt.Key_G, label='G: make an unrelated later merge')
        settle()
        unrelated = int(supervisor.selected[0])
        key(gui, Qt.Key_M, Qt.AltModifier, 'Alt + M: label the unrelated unit MUA')
        settle()
        after_later = supervisor.clustering.spike_clusters.copy()
        before_restore_history = supervisor._global_history.current_position
        unrelated_rows = np.flatnonzero(np.isin(initial, unrelated_inputs))
        report['unrelated'] = {'inputs': unrelated_inputs, 'result': unrelated, 'group': 'mua'}

        def assert_unrelated_preserved():
            np.testing.assert_array_equal(
                supervisor.clustering.spike_clusters[unrelated_rows],
                after_later[unrelated_rows],
            )
            assert supervisor.cluster_meta.get('group', unrelated) == 'mua'

        assert_unrelated_preserved()
        click_row(supervisor.cluster_view, unrelated)
        capture('04-later-edits', 'A later unrelated merge and MUA label must remain intact')

        # Expected scientific memberships come directly from immutable input
        # labels, not the production recursive-lineage resolver being exercised.
        original_groups = {unit: np.flatnonzero(initial == unit) for unit in originals}
        assert all(len(rows) for rows in original_groups.values())
        click_row(supervisor.cluster_view, merged)
        table = supervisor.cluster_view
        index = table._proxy_index_for_id(merged)
        table.table_view.scrollTo(index)
        settle(150)
        point = table.table_view.visualRect(index).center()
        restore_menu_errors = []

        def choose_restore_action():
            try:
                menu = app.activePopupWidget()
                assert isinstance(menu, QMenu)
                action = next(
                    action
                    for action in menu.actions()
                    if action.text() == 'Restore parents in Merge Mode'
                )
                assert action.isEnabled()
                click(
                    menu,
                    Qt.LeftButton,
                    Qt.NoModifier,
                    menu.actionGeometry(action).center(),
                    'Restore all merge components in Merge Mode',
                )
            except Exception as error:
                restore_menu_errors.append(error)
                menu = app.activePopupWidget()
                if menu is not None:
                    menu.close()

        input_note('Right click: older nested merge context menu')
        move_pointer(table.table_view.viewport(), point)
        QTimer.singleShot(650, choose_restore_action)
        native_context_click()
        settle(1000)
        if restore_menu_errors:
            raise restore_menu_errors[0]
        assert controller.parent_inspection.window is None
        assert supervisor.selection.state.is_merge_mode
        restored = {}
        for old_parent, rows in original_groups.items():
            values = np.unique(supervisor.clustering.spike_clusters[rows])
            assert len(values) == 1
            restored[old_parent] = int(values[0])
        assert len(set(restored.values())) == 3
        assert merged not in supervisor.clustering.cluster_ids
        assert first_merged not in supervisor.clustering.cluster_ids
        assert set(supervisor.selection.state.merge.ordered_ids) == set(restored.values())
        assert supervisor._global_history.current_position == before_restore_history + 1
        for old_parent, child in restored.items():
            np.testing.assert_array_equal(
                supervisor.clustering.spikes_per_cluster[child], original_groups[old_parent]
            )
            assert supervisor.parent_origins[child] == old_parent
        assert len({supervisor.selection_color_indices[unit] for unit in restored.values()}) == 3
        assert_unrelated_preserved()
        assert_slay_status(False)
        report['restored'] = restored
        report['restored_live_units'] = True
        report['recursive_restore_exact'] = True
        report['one_restore_history_entry'] = True
        report['restored_historical_ids_visible'] = True
        report['restored_color_indices'] = {
            str(old): supervisor.selection_color_indices[new] for old, new in restored.items()
        }
        capture(
            '05-restored-parents',
            'One restoration returns all three original components, not the intermediate merge',
            restored=restored,
            slay_status='accepted_restored',
        )

        restored_assignments = supervisor.clustering.spike_clusters.copy()
        key(gui, Qt.Key_Z, Qt.ControlModifier, 'Ctrl + Z: undo the whole recursive restoration')
        settle()
        np.testing.assert_array_equal(supervisor.clustering.spike_clusters, after_later)
        assert_unrelated_preserved()
        key(
            gui,
            Qt.Key_Z,
            Qt.ControlModifier | Qt.ShiftModifier,
            'Ctrl + Shift + Z: redo the whole recursive restoration',
        )
        settle()
        np.testing.assert_array_equal(supervisor.clustering.spike_clusters, restored_assignments)
        assert supervisor.selection.state.is_merge_mode
        assert_slay_status(False)
        assert_unrelated_preserved()
        capture('06-restore-history', 'One Undo/Redo reverses the entire recursive restoration')

        key(gui, Qt.Key_V, label='V: leave Merge Mode; all three restored clusters stay live')
        settle()
        assert not supervisor.selection.state.is_merge_mode
        np.testing.assert_array_equal(supervisor.clustering.spike_clusters, restored_assignments)
        assert set(restored.values()) <= set(supervisor.clustering.cluster_ids)
        assert_unrelated_preserved()
        capture(
            '07-parents-remain-live', 'The restored source components remain real live clusters'
        )
        key(gui, Qt.Key_V, label='V: stage the restored components again')
        settle()

        excluded = restored[third]
        click_row(supervisor.merge_view, excluded)
        exclude_button = supervisor.merge_view.dock.get_widget('exclude_merge_row')
        assert (
            exclude_button is not None
            and exclude_button.isVisible()
            and exclude_button.isEnabled()
        )
        click(exclude_button, Qt.LeftButton, label='Exclude the third source component')
        settle()
        chosen = [restored[unit] for unit in first_parents]
        assert set(supervisor.selected) == set(chosen)
        assert excluded in supervisor.clustering.cluster_ids
        np.testing.assert_array_equal(supervisor.clustering.spike_clusters, restored_assignments)
        report['excluded_parent'] = {'original': third, 'live': excluded}
        report['chosen'] = chosen
        assert_slay_status(False)
        capture(
            '08-excluded-parent',
            'Exclude the third component and keep the original suggested pair',
        )
        final_parent_groups = {
            unit: supervisor.clustering.spikes_per_cluster[unit].copy() for unit in chosen
        }
        final_parent_colors = {unit: supervisor.selection_color_indices[unit] for unit in chosen}
        key(gui, Qt.Key_G, label='G: remerge the exact original SLAY pair')
        settle()
        final_merged = int(supervisor.selected[0])
        expected = initial.copy()
        expected[np.isin(initial, first_parents)] = final_merged
        expected[initial == third] = excluded
        expected[np.isin(initial, unrelated_inputs)] = unrelated
        np.testing.assert_array_equal(supervisor.clustering.spike_clusters, expected)
        assert_slay_status(True)
        assert_unrelated_preserved()
        assert (
            dict(supervisor.merge_propositions.catalog.reviews[proposal.key].mapping())
            == accepted_review
        )
        report['final_merged'] = final_merged
        report['exact_remerge_slay_green'] = True
        report['historical_review_preserved'] = True
        report['later_edits_preserved'] = True
        report['excluded_parent_stays_independent'] = True
        capture('09-exact-remerge', 'The exact SLAY pair is merged again: its suggestion is green')

        key(gui, Qt.Key_Z, Qt.ControlModifier, 'Ctrl + Z: undo the remerge; SLAY becomes white')
        settle()
        np.testing.assert_array_equal(supervisor.clustering.spike_clusters, restored_assignments)
        assert_slay_status(False)
        assert_unrelated_preserved()
        key(
            gui,
            Qt.Key_Z,
            Qt.ControlModifier | Qt.ShiftModifier,
            'Ctrl + Shift + Z: redo the remerge; SLAY becomes green',
        )
        settle()
        np.testing.assert_array_equal(supervisor.clustering.spike_clusters, expected)
        assert_slay_status(True)
        assert_unrelated_preserved()
        report['remerge_undo_redo_status_correct'] = True
        capture(
            '10-remerge-history',
            'Undo/Redo keeps the SLAY color synchronized with actual membership',
        )

        if bounded:
            assert len(controller.bounded_session.events) == len(initial_events) + 6
        key(gui, Qt.Key_S, Qt.ControlModifier, 'Ctrl + S: save the revised curation')
        settle()
        report['assigned_spikes'] = int(np.count_nonzero(initial != expected))
        capture(
            '11-revised-save',
            'Saved: remerged pair, independent third component and later MUA edit',
        )
        input_note('Reopen the saved working copy')
        gui.close()
        controller.close()
        gui.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        app.processEvents()
        gui = controller = None
        controller = TemplateController(
            model=None if bounded else load_model(args.dataset / 'params.py'),
            dir_path=args.dataset,
            config_dir=args.output / 'config',
            plugins=[],
            enable_threading=False,
        )
        controller.supervisor.columns = display_columns
        gui = controller.create_gui(
            default_views=names, view_count=dict.fromkeys(names, 1), do_prompt_save=False
        )
        if hasattr(controller, '_summary_store_actions'):
            controller._summary_store_actions._checked = True
        supervisor = controller.supervisor
        propositions = supervisor.merge_propositions_view
        gui.resize(1920, 1200)
        gui.move(0, 0)
        gui.show()
        supervisor.select([final_merged])
        settle()
        if bounded:
            report['reopened_initial_evidence_window'] = controller.evidence_window_index

            def choose_evidence_window():
                dialog = app.activeModalWidget()
                assert isinstance(dialog, QInputDialog)
                edit = dialog.findChild(QLineEdit)
                assert edit is not None and edit.isVisible()
                input_note('Type evidence window index, then Enter')
                move_pointer(edit, edit.rect().center())
                edit.selectAll()
                QTest.keyClicks(edit, str(report['evidence_window']), delay=120)
                key(
                    edit,
                    Qt.Key_Return,
                    label=f'Enter: inspect evidence window {report["evidence_window"]}',
                )

            QTimer.singleShot(800, choose_evidence_window)
            input_note('View menu: Inspect evidence window')
            gui.view_actions.get('inspect_evidence_window').trigger()
            settle()
            assert controller.evidence_window_index == report['evidence_window']
            report['reopened_selected_evidence_window'] = controller.evidence_window_index
            report['reopen_window_action'] = (
                'Explicit Inspect evidence window dialog; not automatic restoration'
            )
        np.testing.assert_array_equal(expected, supervisor.clustering.spike_clusters)
        assert_unrelated_preserved()
        groups = controller.parent_inspection.lineage.parents(final_merged, recursive=False)
        assert set(groups) == set(chosen)
        for parent in chosen:
            np.testing.assert_array_equal(groups[parent], final_parent_groups[parent])
        assert (
            dict(controller.parent_inspection.lineage.colors(final_merged, groups))
            == final_parent_colors
        )
        assert_slay_status(True)
        assert (
            dict(supervisor.merge_propositions.catalog.reviews[proposal.key].mapping())
            == accepted_review
        )
        report['reopened_parent_colors_preserved'] = True
        report['reopened_lineage_correct'] = True
        report['reopened_slay_green'] = True
        report['historical_review_statuses'] = {
            proposal.key: supervisor.merge_propositions.catalog.status_for(proposal.key).value
        }
        capture(
            '12-reopened', 'Reopened: green SLAY pair, independent third component and later edits'
        )
        assert {p.name: hash_file(p) for p in files} == source_hashes
        report['source_unchanged'] = True
        report['success'] = True
        (args.output / 'capture.json').write_text(json.dumps(report, indent=2))
    finally:
        if recording is not None:
            recording.terminate()
            recording.wait(timeout=30)
        if gui is not None:
            gui.close()
        if controller is not None:
            controller.close()
        if app is not None:
            from PyQt5 import sip

            for widget in app.topLevelWidgets():
                widget.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            app.processEvents()
            sip.delete(app)
        server.terminate()
        server.wait(timeout=10)
        server_log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mode', required=True, choices=['ordinary', 'bounded'])
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()
    if not 1 <= args.fps <= 60:
        parser.error('--fps must be between 1 and 60.')
    run(args)
