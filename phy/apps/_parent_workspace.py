"""An embedded read-only comparison workspace with reversible GUI ownership."""

from contextlib import suppress

from phylib.utils import unconnect

from phy.gui import GUI
from phy.gui.qt import (
    QAbstractItemView,
    QAction,
    QApplication,
    QColor,
    QDockWidget,
    QEvent,
    QHeaderView,
    QKeySequence,
    QLabel,
    QObject,
    Qt,
    QTableWidget,
    QTableWidgetItem,
)
from phy.utils.color import selected_cluster_color

_DARK_STYLE = """
QMainWindow, QToolBar, QDockWidget, QTableWidget { background: #171b22; color: #edf0f5; }
QToolBar { border: 0; padding: 5px; spacing: 12px; }
QToolButton { color: #f5f7fa; background: #303945; border: 1px solid #526174;
              border-radius: 4px; padding: 7px 11px; }
QToolButton:hover, QToolButton:focus { background: #425166; }
QHeaderView::section { color: #d7dce4; background: #242b35; border: 0; padding: 5px; }
QTableWidget { border: 0; gridline-color: #2e3642; }
QDockWidget::title { color: #d7dce4; background: #242b35; padding: 4px; }
QScrollBar { background: #202630; }
"""


class _ReviewGUI(GUI):
    """A Qt.Widget child, with no independent window or persistent close path."""

    def closeEvent(self, event):
        event.accept()
        self._return_to_unit()


class ParentWorkspace(QObject):
    """Own the temporary host dock, keyboard guard and exact restoration snapshot."""

    def __init__(
        self, host, supervisor, target, groups, counts, colors, bounded, config_dir, on_back
    ):
        super().__init__(host)
        self.host, self.supervisor, self.on_back = host, supervisor, on_back
        self.active = True
        self.original_state = host.saveState()
        self.original_focus = QApplication.focusWidget()
        self.original_docks = [
            d for d in host.findChildren(QDockWidget) if d.parentWidget() is host
        ]
        self.visibility = {d: not d.isHidden() for d in self.original_docks}
        self.action_states = {a: a.isEnabled() for a in host.findChildren(QAction)}
        self._action_guards = []
        self._blocked_sequences = []

        try:
            self._build(target, groups, counts, colors, bounded, config_dir, on_back)
        except Exception:
            # Construction has not touched the host layout/actions yet, but the
            # child GUI already owns global event-registry callbacks.
            if hasattr(self, 'gui'):
                unconnect(self.gui)
                self.gui.setParent(None)
                self.gui.deleteLater()
            if hasattr(self, 'host_dock'):
                self.host_dock.deleteLater()
            self.active = False
            self.deleteLater()
            raise

    def _build(self, target, groups, counts, colors, bounded, config_dir, on_back):
        host = self.host
        self.gui = _ReviewGUI(
            name='ParentReview',
            subtitle=f'Parents of {target}',
            config_dir=config_dir,
            default_state_path=None,
        )
        self.gui.setParent(host, Qt.Widget)
        self.gui._return_to_unit = on_back
        self.gui.menuBar().hide()
        self.gui.statusBar().hide()
        self.gui.setStyleSheet(_DARK_STYLE)
        self.gui._toolbar.setMovable(False)
        self.gui._toolbar.setFloatable(False)
        self.gui._toolbar.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.gui.file_actions.add(
            on_back, name='return_to_merged_unit', docstring='Back to merged unit'
        )
        back = self.gui.file_actions.get('return_to_merged_unit')
        back.setText('Back to merged unit')
        self.gui._toolbar.addAction(back)
        self.return_button = self.gui._toolbar.widgetForAction(back)
        self.return_button.setObjectName('parent-review-return')
        title = QLabel(f'Parents of {target}   ·   Read only   ·   Esc to return')
        title.setStyleSheet('color: #edf0f5; background: transparent; font-size: 13px;')
        self.gui._toolbar.addWidget(title)
        self.gui._toolbar.show()

        self.parent_table = QTableWidget(len(groups), 4, self.gui)
        table = self.parent_table
        table.setObjectName('parent-review-table')
        table.setHorizontalHeaderLabels(
            ('', 'Parent', 'Retained' if bounded else 'Spikes', 'Full')
        )
        table.verticalHeader().hide()
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.NoSelection)
        table.setFocusPolicy(Qt.NoFocus)
        table.setWordWrap(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)
        for row, (parent, ids) in enumerate(groups.items()):
            color = QColor(
                *(round(float(c) * 255) for c in selected_cluster_color(colors[parent])[:3])
            )
            full = counts.get(parent)
            values = ('■', str(parent), f'{len(ids):,}', '—' if full is None else f'{full:,}')
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(Qt.ItemIsEnabled)
                item.setForeground(color if column == 0 else QColor('#edf0f5'))
                item.setTextAlignment(
                    Qt.AlignCenter if column == 0 else Qt.AlignRight | Qt.AlignVCenter
                )
                item.setToolTip(
                    f'{target} ← {parent}'
                    + (
                        '; full-recording count unavailable'
                        if full is None
                        else f'; {full:,} full spikes'
                    )
                )
                table.setItem(row, column, item)
        table.setColumnHidden(3, not bounded)
        table.resizeColumnsToContents()
        self.table_dock = QDockWidget('Immediate parents', self.gui)
        self.table_dock.setObjectName('ParentReviewTable')
        self.table_dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.table_dock.setWidget(table)
        self.table_dock.setMinimumWidth(218)
        self.table_dock.setMaximumWidth(280)
        self.gui.addDockWidget(Qt.LeftDockWidgetArea, self.table_dock)
        self.scope = QLabel(
            'Retained windows only; full counts when known.'
            if bounded
            else 'Exact parent membership; plots use display sampling.'
        )
        self.scope.setWordWrap(True)
        self.scope.setStyleSheet('color: #c4ccd7; background: #171b22; padding: 4px;')
        self.gui.statusBar().addWidget(self.scope, 1)
        self.gui.statusBar().show()

        self.host_dock = QDockWidget(host)
        self.host_dock.setObjectName('ParentReviewWorkspace')
        self.host_dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.host_dock.setTitleBarWidget(QLabel(''))
        self.host_dock.setWidget(self.gui)

    def _activate(self):
        for dock in self.original_docks:
            if dock is not self.supervisor.cluster_view.dock:
                dock.hide()
        self.host.addDockWidget(Qt.RightDockWidgetArea, self.host_dock)
        self.host.resizeDocks(
            [self.supervisor.cluster_view.dock, self.host_dock],
            [220, max(600, self.host.width() - 230)],
            Qt.Horizontal,
        )

        navigation = {'select', 'next', 'previous', 'first', 'last', 'next_best', 'previous_best'}
        allowed = {
            group.get(name)
            for group in self.host.actions
            if group is self.supervisor.select_actions
            for name in navigation
            if group.get(name) is not None
        }
        for action in self.action_states:
            if action in allowed or action.text().replace('&', '').lower() in ('exit', 'about'):
                continue
            self._blocked_sequences.extend(s for s in action.shortcuts() if not s.isEmpty())

            def guard(action=action):
                if self.active and action.isEnabled():
                    action.setEnabled(False)

            action.changed.connect(guard)
            self._action_guards.append((action, guard))
            action.setEnabled(False)
        QApplication.instance().installEventFilter(self)

    def show(self):
        self._activate()
        self.host_dock.show()
        self.gui.show()
        self.host_dock.raise_()
        self.return_button.setFocus(Qt.OtherFocusReason)

    def eventFilter(self, obj, event):
        if not self.active:
            return False
        if obj is self.host and event.type() == QEvent.Close:
            self.on_back()  # Before GUI.closeEvent can save the temporary dock state.
            return False
        if event.type() not in (QEvent.KeyPress, QEvent.ShortcutOverride):
            return False
        current = obj
        while current is not None and current is not self.host:
            current = current.parent()
        focus = QApplication.focusWidget()
        if current is None and (focus is None or focus.window() is not self.host):
            return False
        if event.key() == Qt.Key_Escape:
            event.accept()
            if event.type() == QEvent.KeyPress:
                self.on_back()
            return True
        sequence = QKeySequence(int(event.modifiers()) | event.key())
        if event.text() == ':' or any(
            s.matches(sequence) == QKeySequence.ExactMatch for s in self._blocked_sequences
        ):
            event.accept()
            return True
        return False

    def restore(self):
        if not self.active:
            return
        self.active = False  # Visibility/focus changes may synchronously publish selections.
        QApplication.instance().removeEventFilter(self)
        self.host_dock.hide()
        self.host.removeDockWidget(self.host_dock)
        self.host_dock.setParent(None)
        self.host.restoreState(self.original_state)
        for dock, visible in self.visibility.items():
            with suppress(RuntimeError):
                dock.setVisible(visible)
        for action, guard in self._action_guards:
            with suppress(RuntimeError, TypeError):
                action.changed.disconnect(guard)
        for action, enabled in self.action_states.items():
            with suppress(RuntimeError):
                action.setEnabled(enabled)
        focus_restored = False
        if self.original_focus is not None:
            with suppress(RuntimeError):
                self.original_focus.setFocus(Qt.OtherFocusReason)
                focus_restored = True
        if not focus_restored:
            with suppress(RuntimeError):
                self.supervisor.cluster_view.table_view.setFocus(Qt.OtherFocusReason)
        self.host_dock.deleteLater()
        self.deleteLater()
