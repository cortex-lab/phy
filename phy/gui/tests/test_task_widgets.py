"""Task presentation and cleanup tolerate chained runs and destroyed Qt owners."""

from threading import Event

import pytest
from PyQt5 import sip

from phy.gui.task_widgets import TaskWidget
from phy.gui.tasks import BackgroundTaskManager, TaskCancelledError, TaskProgress, TaskSpec


def test_completed_task_can_close_reopen_details_and_start_another_run(qtbot):
    manager = BackgroundTaskManager()
    widget = TaskWidget(manager)
    qtbot.addWidget(widget)
    started, release = Event(), Event()

    def worker(request, context):
        context.report(TaskProgress('Reading rows', 1, 2, message='Rows checked'))

    def next_worker(request, context):
        started.set()
        assert release.wait(5)

    try:
        manager.submit(TaskSpec('widget-first-run', 'First run', worker, None))
        qtbot.waitUntil(lambda: not manager.busy)
        assert widget.label.text() == 'Done'
        assert widget.bar.isHidden()
        assert widget.cancel_button.isHidden()
        assert widget.hide_button.text() == 'Close'
        widget.hide_button.click()
        assert widget.isHidden()
        widget.show()
        widget.details_button.click()
        assert widget.details.isVisible()
        assert 'Rows checked' in widget.log.toPlainText()
        widget.details.close()

        widget.notice('Store created; finalization needs attention. See Details.')
        assert 'needs attention' in widget.label.text()
        assert widget.bar.isHidden()
        manager.submit(TaskSpec('widget-next-run', 'Next run', next_worker, None))
        qtbot.waitUntil(started.is_set)
        assert widget.label.text() == 'Next run'
        assert widget.label.toolTip() == ''
        assert not widget.bar.isHidden()
        assert not widget.cancel_button.isHidden()
        assert widget.cancel_button.isEnabled()
        assert widget.hide_button.text() == 'Hide'
    finally:
        release.set()
        qtbot.waitUntil(lambda: not manager.busy)
        widget.dispose()
        manager.detach()


@pytest.mark.parametrize('state', ['cancelled', 'failed'])
def test_unsuccessful_task_has_clear_terminal_controls_and_retains_error(qtbot, state):
    manager = BackgroundTaskManager()
    widget = TaskWidget(manager)
    qtbot.addWidget(widget)

    def worker(request, context):
        if state == 'cancelled':
            raise TaskCancelledError()
        raise ValueError('Could not read source')

    try:
        manager.submit(TaskSpec('widget-unsuccessful', 'Reading source', worker, None))
        qtbot.waitUntil(lambda: not manager.busy)
        assert manager.state == state
        assert widget.label.text() == ('Cancelled' if state == 'cancelled' else 'Task failed')
        assert widget.bar.isHidden()
        assert widget.cancel_button.isHidden()
        assert widget.hide_button.text() == 'Close'
        if state == 'failed':
            assert widget.estimate.text() == 'See Details'
            assert 'Could not read source' in widget.log.toPlainText()
    finally:
        widget.dispose()
        manager.detach()


@pytest.mark.parametrize('order', ['manager-first', 'widget-first', 'completed-manager-first'])
def test_dispose_is_safe_and_repeatable_after_qt_deletion(qtbot, order):
    manager = BackgroundTaskManager()
    widget = TaskWidget(manager)
    if order == 'completed-manager-first':
        manager.submit(TaskSpec('widget-complete', 'Complete', lambda r, c: None, None))
        qtbot.waitUntil(lambda: not manager.busy)
    if order == 'widget-first':
        sip.delete(widget)
    else:
        sip.delete(manager)
    widget.dispose()
    widget.dispose()
    manager.detach()
    assert widget._disposed
    assert manager._detached
    if not sip.isdeleted(widget):
        widget.close()
    if not sip.isdeleted(manager):
        sip.delete(manager)
