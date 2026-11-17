"""Exercise Carl's feedback in the real GUI on a disposable data copy.

Starts its own Xvfb server: no window or recording reaches the desktop. Inputs
must be disposable copies; curation is deliberately saved. Records real X11
pixels, Qt input events and state assertions, with a separate per-run config.
"""

import argparse
import json
import os
import shutil
import subprocess
import time
from copy import deepcopy
from pathlib import Path


def run(args):
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
        from PyQt5.QtGui import QContextMenuEvent
        from PyQt5.QtTest import QTest
        from PyQt5.QtWidgets import QMenu

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
            'capture_scope': 'Settled real X11 pixels; Qt input events; not a spontaneous-redraw test',
            'private_display': display,
            'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'steps': [],
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

        def settle(ms=1000):
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
                '12',
                '-f',
                'x11grab',
                '-draw_mouse',
                '0',
                '-i',
                display + '.0',
                '-an',
                '-c:v',
                'libx264',
                '-preset',
                'veryfast',
                '-crf',
                '24',
                '-pix_fmt',
                'yuv420p',
                '-movflags',
                '+faststart',
                str(args.output / 'walkthrough.mp4'),
            ],
            stdout=video_log,
            stderr=video_log,
        )

        def capture(name, title, window=None, **facts):
            target = window or gui
            if window is None:
                gui.resize(1920, 1200)
                gui.move(0, 0)
            gui.statusBar().showMessage(title)
            settle(1400)
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
            QTest.mouseClick(
                table.table_view.viewport(),
                Qt.LeftButton,
                modifiers,
                table.table_view.visualRect(index).center(),
            )
            settle()

        catalog = supervisor.merge_propositions.catalog
        choices = [
            p
            for p in catalog.propositions
            if all(unit in supervisor.clustering.cluster_ids for unit in p.unit_ids)
        ]
        if bounded:
            desired = [(40, 44, 48), (49, 50, 51, 53)]
            choices = [next(p for p in choices if tuple(p.unit_ids) == ids) for ids in desired]
        else:
            choices = choices[:2]
        assert len(choices) >= 2
        a, b = choices[:2]
        keys = (a.key, b.key)
        union = list(dict.fromkeys((*a.unit_ids, *b.unit_ids)))
        report['propositions'] = [{'key': p.key, 'unit_ids': p.unit_ids} for p in choices[:2]]
        report['union'] = union
        if bounded:
            # Pick a recorded evidence window containing stored waveforms for
            # these units, rather than displaying an unrelated empty window.
            stored = controller.model.spike_waveforms.spike_ids
            labels = supervisor.clustering.spike_clusters[stored]
            slots = np.searchsorted(
                controller.evidence_windows[:, 1],
                controller.model.spike_samples[stored],
                side='right',
            )
            counts = np.stack(
                [
                    np.bincount(slots[labels == unit], minlength=len(controller.evidence_windows))
                    for unit in union
                ]
            )
            score = (counts > 0).sum(axis=0) * 10000 + counts.sum(axis=0)
            window_index = int(np.argmax(score))
            controller.inspect_evidence_window(window_index)
            amplitude.amplitude_scope = 'overview'
            report['evidence_window'] = window_index
            report['window_waveforms_by_unit'] = dict(
                zip(map(str, union), map(int, counts[:, window_index]))
            )
        initial = supervisor.clustering.spike_clusters.copy()
        initial_reviews = dict(catalog.reviews)
        initial_events = list(controller.bounded_session.events) if bounded else None
        for i, unit in enumerate(a.unit_ids[:3]):
            click_row(supervisor.cluster_view, unit, Qt.ControlModifier if i else Qt.NoModifier)
        capture('01-initial', 'Real-data cluster comparison before the new workflow')

        # Actual right-click on a scientific native OpenGL child window.
        transfer = int(ccg.cluster_ids[1])
        n = len(ccg.cluster_ids)
        ndc = np.array([[-1 + 1 / n, 1 - 3 / n]])
        ndc = ccg._display_scale.apply(ndc)[0]
        width, height = ccg.canvas.get_size()
        pos = QPoint(round((ndc[0] + 1) * width / 2), round((1 - ndc[1]) * height / 2))
        QTest.mouseClick(ccg.canvas, Qt.RightButton, Qt.NoModifier, pos)
        settle()
        assert transfer in supervisor.selected_similar, (transfer, supervisor.selected_similar)
        capture(
            '02-right-click-transfer',
            'Right-click transfers the correlogram row to Similarity',
            transferred=transfer,
        )

        propositions = supervisor.merge_propositions_view
        click_row(propositions, propositions._id_by_key[a.key])
        capture('03-single-suggestion', 'One SLAY suggestion in the Merge workspace')
        click_row(propositions, propositions._id_by_key[b.key], Qt.ControlModifier)
        assert propositions.selected_keys == keys
        assert supervisor.selected == union, (supervisor.selected, union)
        np.testing.assert_array_equal(initial, supervisor.clustering.spike_clusters)
        assert dict(supervisor.merge_propositions.catalog.reviews) == initial_reviews
        capture(
            '04-control-click-union',
            'Ctrl-click combines two suggestions; assignments unchanged',
            selected_keys=list(keys),
            union=union,
        )
        click_row(propositions, propositions._id_by_key[b.key], Qt.ControlModifier)
        assert propositions.selected_keys == (a.key,)
        click_row(propositions, propositions._id_by_key[b.key], Qt.ControlModifier)
        assert supervisor.selected == union
        # Native Qt shortcut, with the application window active.
        gui.activateWindow()
        app.setActiveWindow(gui)
        gui.setFocus()
        settle(150)
        QTest.keyClick(gui, Qt.Key_G)
        settle()
        assert len(supervisor.selected) == 1, supervisor.selected
        merged = int(supervisor.selected[0])
        expected = initial.copy()
        expected[np.isin(initial, union)] = merged
        np.testing.assert_array_equal(expected, supervisor.clustering.spike_clusters)
        assert all(
            supervisor.merge_propositions.catalog.reviews[key].result_unit_id == merged
            for key in keys
        )
        capture('05-merged', 'G commits one merge and both suggestion decisions', merged=merged)

        # Snapshot before opening, including history/metadata/dirty state.
        before_review = supervisor.clustering.spike_clusters.copy()
        before_metadata = deepcopy(supervisor.cluster_meta._data)
        before_dirty = supervisor.is_dirty()
        before_history = deepcopy(supervisor._global_history.current_position)
        # Open the actual Cluster View context menu and click Show parents.
        table = supervisor.cluster_view
        index = table._proxy_index_for_id(merged)
        table.table_view.scrollTo(index)
        settle(150)
        point = table.table_view.visualRect(index).center()

        def choose_context_action():
            menu = app.activePopupWidget()
            assert isinstance(menu, QMenu)
            action = next(action for action in menu.actions() if action.text() == 'Show parents')
            assert action.isEnabled()
            QTest.mouseClick(
                menu, Qt.LeftButton, Qt.NoModifier, menu.actionGeometry(action).center()
            )

        QTimer.singleShot(1000, choose_context_action)
        event = QContextMenuEvent(
            QContextMenuEvent.Mouse, point, table.table_view.viewport().mapToGlobal(point)
        )
        QApplication.postEvent(table.table_view.viewport(), event)
        settle()
        review = controller.parent_inspection.window
        assert review is not None
        review.resize(1880, 1120)
        review.move(20, 20)
        settle()
        np.testing.assert_array_equal(before_review, supervisor.clustering.spike_clusters)
        np.testing.assert_equal(supervisor.cluster_meta._data, before_metadata)
        assert supervisor.is_dirty() == before_dirty
        assert supervisor._global_history.current_position == before_history
        assert review._toolbar.isVisible()
        capture(
            '06-show-parents',
            'Exact immediate merge parents — read only',
            window=review,
            parents=list(map(int, controller.parent_inspection.providers.ids)),
        )
        button = review._toolbar.widgetForAction(review.file_actions.get('return_to_merged_unit'))
        assert button.isVisible() and button.isEnabled()
        assert review.rect().contains(button.mapTo(review, button.rect().center()))
        QTest.mouseClick(button, Qt.LeftButton)
        settle()
        assert controller.parent_inspection.window is None
        np.testing.assert_array_equal(before_review, supervisor.clustering.spike_clusters)
        np.testing.assert_equal(supervisor.cluster_meta._data, before_metadata)
        assert supervisor.is_dirty() == before_dirty
        assert supervisor._global_history.current_position == before_history
        capture('07-return', 'One click returns to the merged unit without editing')
        app.setActiveWindow(gui)
        gui.setFocus()
        QTest.keyClick(gui, Qt.Key_Z, Qt.ControlModifier)
        settle()
        np.testing.assert_array_equal(initial, supervisor.clustering.spike_clusters)
        assert supervisor.selection.state.merge.proposition_ids == keys
        assert dict(supervisor.merge_propositions.catalog.reviews) == initial_reviews
        capture(
            '08-undo', 'Undo restores assignments, both decisions and the comparison workspace'
        )
        app.setActiveWindow(gui)
        QTest.keyClick(gui, Qt.Key_Z, Qt.ControlModifier | Qt.ShiftModifier)
        settle()
        assert supervisor.selected == [merged]
        np.testing.assert_array_equal(expected, supervisor.clustering.spike_clusters)
        QTest.keyClick(gui, Qt.Key_S, Qt.ControlModifier)
        settle()
        capture('09-redo-save', 'Redo and Save retain the combined merge')
        if bounded:
            assert len(controller.bounded_session.events) == len(initial_events) + 1
        report['merged'] = merged
        report['assigned_spikes'] = int(
            np.count_nonzero(initial != supervisor.clustering.spike_clusters)
        )
        gui.close()
        controller.close()
        gui.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        app.processEvents()
        gui = controller = None
        # Reopen the same working copy with the saved curation and lineage.
        controller = TemplateController(
            model=None if bounded else load_model(args.dataset / 'params.py'),
            dir_path=args.dataset,
            config_dir=args.output / 'config',
            plugins=[],
            enable_threading=False,
        )
        gui = controller.create_gui(
            default_views=names, view_count=dict.fromkeys(names, 1), do_prompt_save=False
        )
        if hasattr(controller, '_summary_store_actions'):
            controller._summary_store_actions._checked = True
        supervisor = controller.supervisor
        gui.resize(1920, 1200)
        gui.move(0, 0)
        gui.show()
        supervisor.select([merged])
        settle()
        assert all(
            supervisor.merge_propositions.catalog.reviews[key].result_unit_id == merged
            for key in keys
        )
        np.testing.assert_array_equal(expected, supervisor.clustering.spike_clusters)
        assert set(controller.parent_inspection.lineage.parents(merged)) == set(union)
        capture(
            '10-reopened',
            'Reopened: saved assignments, review decisions and parent history verified',
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
    run(parser.parse_args())
