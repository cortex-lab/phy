"""Task presentation cleanup tolerates independently destroyed Qt owners."""

import pytest
from PyQt5 import sip

from phy.gui.task_widgets import TaskWidget
from phy.gui.tasks import BackgroundTaskManager, TaskSpec


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
