"""Qt widgets for GUIs."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import ast
import inspect
import json
import logging
import re
from functools import partial

from phylib.utils import connect, emit
from phylib.utils._misc import _CustomEncoder, _pretty_floats
from phylib.utils._types import _is_integer
from qtconsole.inprocess import QtInProcessKernelManager
from qtconsole.rich_jupyter_widget import RichJupyterWidget

from phy.utils.color import _is_bright, colormaps

from .qt import (
    Debouncer,
    QAbstractItemView,
    QAbstractTableModel,
    QApplication,
    QBrush,
    QCheckBox,
    QColor,
    QDoubleSpinBox,
    QEvent,
    QGridLayout,
    QHeaderView,
    QItemSelectionModel,
    QLabel,
    QLineEdit,
    QModelIndex,
    QPalette,
    QPlainTextEdit,
    QSize,
    QSortFilterProxyModel,
    QSpinBox,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    Qt,
    QTableView,
    QTimer,
    QVBoxLayout,
    QWidget,
    _block,
)

logger = logging.getLogger(__name__)
_NO_VALUE = object()


# -----------------------------------------------------------------------------
# IPython widget
# -----------------------------------------------------------------------------


def _ensure_async_ipykernel_methods(kernel):
    """Adapt synchronous in-process kernel handlers to the newer awaitable API."""
    if kernel is None:  # pragma: no cover
        return

    do_history = getattr(kernel, 'do_history', None)
    if do_history is not None and not inspect.iscoroutinefunction(do_history):

        async def do_history_async(*args, **kwargs):
            shell = getattr(kernel, 'shell', None)
            if getattr(shell, 'history_manager', None) is None:
                return {'status': 'ok', 'history': []}
            return do_history(*args, **kwargs)

        kernel.do_history = do_history_async


class IPythonView(RichJupyterWidget):
    """A view with an IPython console living in the same Python process as the GUI."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_manager = None
        self.kernel_client = None
        self.kernel = None
        self.shell = None

    def start_kernel(self):
        """Start the IPython kernel."""

        logger.debug('Starting the kernel.')

        self.kernel_manager = QtInProcessKernelManager()
        self.kernel_manager.start_kernel(show_banner=False)
        self.kernel_manager.kernel.gui = 'qt'
        self.kernel = self.kernel_manager.kernel
        _ensure_async_ipykernel_methods(self.kernel)
        self.shell = self.kernel.shell
        # This embedded shell should not persist readline history during tests or GUI teardown.
        history_manager = getattr(self.shell, 'history_manager', None)
        if history_manager is not None:
            history_manager.enabled = False

        try:
            self.kernel_client = self.kernel_manager.client()
            self.kernel_client.start_channels()
        except Exception as e:  # pragma: no cover
            logger.error('Could not start IPython kernel: %s.', str(e))

        self.set_default_style('linux')
        self.exit_requested.connect(self.stop)

    def inject(self, **kwargs):
        """Inject variables into the IPython namespace."""
        logger.debug('Injecting variables into the kernel: %s.', ', '.join(kwargs.keys()))
        try:
            self.kernel.shell.push(kwargs)
        except Exception as e:  # pragma: no cover
            logger.error('Could not inject variables to the IPython kernel: %s.', str(e))

    def attach(self, gui, **kwargs):
        """Add the view to the GUI, start the kernel, and inject the specified variables."""
        gui.add_view(self)
        self.start_kernel()
        self.inject(gui=gui, **kwargs)
        try:
            import numpy

            self.inject(np=numpy)
        except ImportError:  # pragma: no cover
            pass
        try:
            import matplotlib.pyplot as plt

            self.inject(plt=plt)
        except ImportError:  # pragma: no cover
            pass

        @connect(sender=self)
        def on_close_view(view, gui):
            self.stop()

    def stop(self):
        """Stop the kernel."""
        logger.debug('Stopping the kernel.')
        kernel = self.kernel

        if self.kernel_client is not None:
            try:
                self.kernel_client.stop_channels()
            except Exception as e:  # pragma: no cover
                logger.error('Could not stop IPython kernel channels: %s.', str(e))
            self.kernel_client = None

        if kernel is not None:
            shell = getattr(kernel, 'shell', None)
            if shell is not None:
                try:
                    shell._atexit_once()
                except Exception as e:  # pragma: no cover
                    logger.error('Could not finalize IPython shell cleanup: %s.', str(e))

            for stream_name in ('stdout', 'stderr'):
                stream = getattr(kernel, stream_name, None)
                if stream is None:
                    continue
                try:
                    stream.close()
                except Exception as e:  # pragma: no cover
                    logger.error(
                        'Could not close IPython kernel %s stream: %s.',
                        stream_name,
                        str(e),
                    )

            iopub_thread = getattr(kernel, 'iopub_thread', None)
            if iopub_thread is not None and getattr(iopub_thread, 'thread', None):
                try:
                    iopub_thread.stop()
                except Exception as e:  # pragma: no cover
                    logger.error('Could not stop IPython IOPub thread: %s.', str(e))

        if self.kernel_manager is not None:
            try:
                self.kernel_manager.shutdown_kernel()
            except Exception as e:  # pragma: no cover
                logger.error('Could not shut down IPython kernel: %s.', str(e))
            self.kernel_manager = None

        self.kernel = None
        self.shell = None

    def closeEvent(self, event):
        self.stop()
        super().closeEvent(event)


def _uniq(seq):
    """Return the list of unique integers in a sequence, by keeping the order."""
    seen = set()
    seen_add = seen.add
    return [int(x) for x in seq if not (x in seen or seen_add(x))]


class Barrier:
    """Implement a synchronization barrier."""

    def __init__(self):
        self._keys = []
        self._results = {}
        self._callback_after_all = None

    def _callback(self, key, *args, **kwargs):
        self._results[key] = (args, kwargs)
        if self._callback_after_all and self.have_all_finished():
            self._callback_after_all()

    def __call__(self, key):
        self._keys.append(key)
        return partial(self._callback, key)

    def have_all_finished(self):
        """Whether all tasks have finished."""
        return set(self._keys) == set(self._results.keys())

    def wait(self):
        """Wait until all tasks have finished."""
        _block(self.have_all_finished)

    def after_all_finished(self, callback):
        """Specify the callback function to call after all tasks have finished."""
        self._callback_after_all = callback

    def result(self, key):
        """Return the result of a task specified by its key."""
        return self._results.get(key, None)


def dumps(o):
    """Dump a JSON object into a string, with pretty floats."""
    return json.dumps(_pretty_floats(o), cls=_CustomEncoder)


def _color_styles():
    """Use colormap colors in table widget."""
    return '\n'.join(
        f"""
        #table .color-{i} > td[class='id'] {{
            background-color: rgb({r}, {g}, {b});
            {'color: #000 !important;' if _is_bright((r, g, b)) else ''}
        }}
        """
        for i, (r, g, b) in enumerate(colormaps.default * 255)
    )


class _TableFilterValidator(ast.NodeVisitor):
    """Validate the supported filter-expression subset."""

    _allowed_compare_ops = (
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
    )

    def __init__(self, allowed_names):
        self.allowed_names = set(allowed_names)

    def visit_Expression(self, node):
        self.visit(node.body)

    def visit_BoolOp(self, node):
        if not isinstance(node.op, (ast.And, ast.Or)):
            raise ValueError
        for value in node.values:
            self.visit(value)

    def visit_UnaryOp(self, node):
        if not isinstance(node.op, ast.Not):
            raise ValueError
        self.visit(node.operand)

    def visit_Compare(self, node):
        self.visit(node.left)
        for op in node.ops:
            if not isinstance(op, self._allowed_compare_ops):
                raise ValueError
        for comparator in node.comparators:
            self.visit(comparator)

    def visit_Name(self, node):
        if node.id not in self.allowed_names:
            raise ValueError

    def visit_Constant(self, node):
        return

    def generic_visit(self, node):
        raise ValueError


def _compile_filter_expr(expr, allowed_names):
    """Compile a filter expression into a Python predicate."""
    if not expr:
        return None, False
    translated = re.sub(r'(?<![=!<>])!(?!=)', ' not ', expr)
    translated = translated.replace('&&', ' and ').replace('||', ' or ')
    translated = re.sub(r'\bnull\b', 'None', translated)
    translated = re.sub(r'\btrue\b', 'True', translated)
    translated = re.sub(r'\bfalse\b', 'False', translated)
    tree = ast.parse(translated, mode='eval')
    _TableFilterValidator(allowed_names).visit(tree)
    code = compile(tree, '<table-filter>', 'eval')

    def predicate(row):
        return bool(eval(code, {'__builtins__': {}}, row))

    return predicate, True


class _TableModel(QAbstractTableModel):
    """Model backing the native Qt table."""

    def __init__(self, table):
        super().__init__(table)
        self._table = table
        self._rows = []
        self._rows_by_id = {}

    def rowCount(self, parent=QModelIndex()):  # noqa: B008
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()):  # noqa: B008
        return 0 if parent.isValid() else len(self._table.columns)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal and 0 <= section < len(self._table.columns):
            return self._table.columns[section]
        return section + 1

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        row = self._rows[index.row()]
        column = self._table.columns[index.column()]
        value = row.get(column)

        if role in (Qt.DisplayRole, Qt.EditRole):
            if value is None:
                return ''
            # Qt's model/view cannot display numpy scalars (np.int64/np.float64),
            # which render as blank cells; convert them to native Python types.
            if hasattr(value, 'item'):
                value = value.item()
            return value
        if role == Qt.BackgroundRole and column == 'id':
            color = self._table._selection_background(row.get('id'))
            if color is not None:
                return color
        if role == Qt.ForegroundRole:
            fg = self._table._foreground_color(row, column)
            if fg is not None:
                return fg
        return None

    def set_rows(self, rows):
        self.beginResetModel()
        self._rows = list(rows)
        self._rows_by_id = {row['id']: row for row in self._rows}
        self.endResetModel()

    def row_by_id(self, row_id):
        return self._rows_by_id.get(row_id)

    def ids(self):
        return [row['id'] for row in self._rows]


class _TableProxyModel(QSortFilterProxyModel):
    """Proxy model with typed sorting and expression filtering."""

    def __init__(self, table):
        super().__init__(table)
        self._table = table
        self._predicate = None
        self.setDynamicSortFilter(True)

    def set_filter_predicate(self, predicate):
        self._predicate = predicate
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row, source_parent):
        if self._predicate is None:
            return True
        row = self.sourceModel()._rows[source_row]
        try:
            return self._predicate(row)
        except Exception:
            return True

    def lessThan(self, left, right):
        column = self._table.columns[left.column()]
        left_row = self.sourceModel()._rows[left.row()]
        right_row = self.sourceModel()._rows[right.row()]
        left_value = left_row.get(column)
        right_value = right_row.get(column)
        if left_value is None and right_value is None:
            return False
        if left_value is None:
            return False
        if right_value is None:
            return True
        try:
            return bool(left_value < right_value)
        except TypeError:
            return bool(str(left_value) < str(right_value))


class _TableItemDelegate(QStyledItemDelegate):
    """Custom paint delegate to preserve dark styling and selection colors."""

    def __init__(self, table):
        super().__init__(table)
        self._table = table

    def paint(self, painter, option, index):
        row = self._table._row_for_proxy_index(index)
        if row is None:
            return super().paint(painter, option, index)

        column = self._table.columns[index.column()]
        row_id = row.get('id')
        is_selected = row_id in self._table._selected_ids

        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.state &= ~QStyle.State_HasFocus

        palette = QPalette(opt.palette)
        fg = self._table._foreground_color(row, column)
        bg = None

        if is_selected:
            bg = self._table._selection_background(row_id)
            if fg is None:
                fg = QColor('#ffffff')
        elif fg is None:
            fg = QColor('#ffffff')

        # Paint the selection tint ourselves for every cell in the row so Qt does not fall
        # back to per-cell selected-item styling.
        if bg is not None:
            painter.save()
            painter.fillRect(opt.rect, bg)
            painter.restore()
            opt.backgroundBrush = QBrush()
            opt.state &= ~QStyle.State_Selected

        if bg is not None:
            palette.setColor(QPalette.Base, bg)
            palette.setColor(QPalette.Highlight, bg)
        if fg is not None:
            palette.setColor(QPalette.Text, fg)
            palette.setColor(QPalette.WindowText, fg)
            palette.setColor(QPalette.HighlightedText, fg)
        opt.palette = palette
        super().paint(painter, opt, index)


class Table(QWidget):
    """A sortable native Qt table with a compatibility API for legacy callers."""

    _ready = False

    def __init__(
        self,
        *args,
        columns=None,
        value_names=None,
        data=None,
        sort=None,
        title='',
        debounce_events=(),
    ):
        super().__init__(*args)
        self.setWindowTitle(title)
        self._debouncer = Debouncer()
        self._debounce_events = set(debounce_events)
        self._debouncer.isBusy = False
        self.columns = list(columns or ['id'])
        self.value_names = list(value_names or self.columns)
        self.data = list(data or [])
        self._selected_ids = []
        self._selected_index_offset = 0
        self._filter_text = ''
        self._filter_is_active = False
        self._current_sort = None
        self._no_emit = False
        self._group_colors = {
            'good': QColor('#86D16D'),
            'mua': QColor('#afafaf'),
            'noise': QColor('#777777'),
        }

        self.filter_edit = QLineEdit(self)
        self.filter_edit.setObjectName('table-filter')
        self.filter_edit.returnPressed.connect(self._apply_filter_from_editor)
        self.filter_edit.installEventFilter(self)

        self.table_view = QTableView(self)
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table_view.clicked.connect(self._on_row_clicked)
        self.table_view.horizontalHeader().sectionClicked.connect(self._on_header_clicked)
        self.table_view.verticalHeader().hide()
        self.table_view.setShowGrid(False)
        self.table_view.setAlternatingRowColors(False)
        self.table_view.setFocusPolicy(Qt.NoFocus)
        self.table_view.setWordWrap(False)
        self.table_view.horizontalHeader().setStretchLastSection(False)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self.filter_edit)
        layout.addWidget(self.table_view)

        self._model = _TableModel(self)
        self._proxy = _TableProxyModel(self)
        self._proxy.setSourceModel(self._model)
        self.table_view.setModel(self._proxy)
        self.table_view.setItemDelegate(_TableItemDelegate(self))
        self._apply_dark_style()

        self._init_table(columns=columns, value_names=value_names, data=data, sort=sort)

    @property
    def debouncer(self):
        return self._debouncer

    def eventFilter(self, obj, event):
        if (
            obj is self.filter_edit
            and event.type() == QEvent.KeyPress
            and event.key() == Qt.Key_Escape
        ):
            self.filter_edit.clear()
            self.filter('')
            return True
        return super().eventFilter(obj, event)

    def _normalize_value_names(self, value_names):
        names = []
        for value_name in value_names:
            if isinstance(value_name, str):
                names.append(value_name)
            elif isinstance(value_name, dict):
                names.extend(value_name.get('data', []))
        return names

    def _apply_dark_style(self):
        self.setStyleSheet(
            """
            QWidget {
                background-color: black;
                color: white;
                font-size: 10pt;
            }
            QLineEdit#table-filter {
                background-color: black;
                color: white;
                border: 1px solid #444;
                padding: 5px;
                selection-background-color: #444;
                selection-color: white;
            }
            QTableView {
                background-color: black;
                color: white;
                border: 0;
                outline: 0;
                selection-background-color: transparent;
                selection-color: white;
                gridline-color: #111;
            }
            QTableView::item {
                padding: 4px 6px;
                border: 0;
                background-color: transparent;
            }
            QTableView::item:hover {
                background-color: #222;
            }
            QTableView::item:selected {
                background-color: transparent;
                color: white;
            }
            QHeaderView::section {
                background-color: black;
                color: white;
                border: 0;
                padding: 5px;
            }
            """
        )

    def _init_table(self, columns=None, value_names=None, data=None, sort=None):
        columns = columns or ['id']
        value_names = value_names or columns
        data = data or []

        self.data = list(data)
        self.columns = list(columns)
        self.value_names = list(value_names)
        self._filterable_names = set(self._normalize_value_names(self.value_names))
        self._filterable_names.update(self.columns)

        self._model.set_rows(self.data)
        self._fit_columns()

        emit('pre_build', self)
        connect(event='select', sender=self, func=lambda *args: self.update(), last=True)
        connect(event='ready', sender=self, func=lambda *args: self._set_ready())

        if sort and sort[0]:
            self.sort_by(*sort)
        self._refresh_selection()
        self._schedule_callback(lambda: emit('ready', self))

    def _set_ready(self):
        self._ready = True

    def is_ready(self):
        return self._ready

    def _schedule_callback(self, callback, value=_NO_VALUE):
        if callback is None:
            return
        if value is _NO_VALUE:
            QTimer.singleShot(0, callback)
        else:
            QTimer.singleShot(0, lambda: callback(value))

    def _emit_event(self, name, payload):
        if name in self._debounce_events:
            self._debouncer.submit(emit, name, self, payload)
        else:
            emit(name, self, payload)

    def add_style(self, style):
        """Append a stylesheet fragment."""
        existing = self.styleSheet()
        self.setStyleSheet(f'{existing}\n{style}' if existing else style)

    def _source_row_for_id(self, row_id):
        ids = self._model.ids()
        try:
            return ids.index(row_id)
        except ValueError:
            return None

    def _proxy_index_for_id(self, row_id, column=0):
        source_row = self._source_row_for_id(row_id)
        if source_row is None:
            return QModelIndex()
        source_index = self._model.index(source_row, column)
        return self._proxy.mapFromSource(source_index)

    def _row_for_proxy_index(self, proxy_index):
        if not proxy_index.isValid():
            return None
        source_index = self._proxy.mapToSource(proxy_index)
        if not source_index.isValid():
            return None
        return self._model._rows[source_index.row()]

    def _fit_columns(self):
        self.table_view.resizeColumnsToContents()
        self.table_view.resizeRowsToContents()

    def _visible_ids(self):
        ids = []
        for row in range(self._proxy.rowCount()):
            proxy_index = self._proxy.index(row, 0)
            source_index = self._proxy.mapToSource(proxy_index)
            ids.append(self._model._rows[source_index.row()]['id'])
        return ids

    def _visible_row_ids(self):
        out = []
        for row in range(self._proxy.rowCount()):
            proxy_index = self._proxy.index(row, 0)
            source_index = self._proxy.mapToSource(proxy_index)
            out.append(self._model._rows[source_index.row()]['id'])
        return out

    def _is_masked_id(self, row_id):
        row = self._model.row_by_id(row_id)
        return bool(row and row.get('is_masked'))

    def _selected_visible_ids(self):
        visible = set(self._visible_ids())
        return [row_id for row_id in self._selected_ids if row_id in visible]

    def _selection_background(self, row_id):
        if row_id not in self._selected_ids:
            return None
        pos = self._selected_ids.index(row_id) + self._selected_index_offset
        colors = list(colormaps.default * 255)
        r, g, b = colors[pos % len(colors)]
        return QColor(int(r), int(g), int(b), 160)

    def _foreground_color(self, row, column):
        if column == 'id' and row.get('id') in self._selected_ids:
            pos = self._selected_ids.index(row.get('id')) + self._selected_index_offset
            colors = list(colormaps.default * 255)
            r, g, b = colors[pos % len(colors)]
            if _is_bright((int(r), int(g), int(b))):
                return QColor('#000000')
        group = row.get('group')
        if group in self._group_colors:
            return self._group_colors[group]
        if row.get('is_masked'):
            return QColor('#888888')
        return None

    def _refresh_selection(self):
        selection_model = self.table_view.selectionModel()
        if selection_model is None:
            return
        selection_model.clearSelection()
        for row_id in self._selected_visible_ids():
            index = self._proxy_index_for_id(row_id)
            if index.isValid():
                selection_model.select(
                    index,
                    QItemSelectionModel.Select | QItemSelectionModel.Rows,
                )
        selection_model.setCurrentIndex(QModelIndex(), QItemSelectionModel.NoUpdate)
        self.table_view.viewport().update()

    def _selected_payload(self, kwargs=None):
        selected = self.get_selected_ids()
        next_id = self.get_sibling_id(selected[-1] if selected else None, 'next')
        return {'selected': selected, 'next': next_id, 'kwargs': kwargs or {}}

    def _emit_selected(self, kwargs=None):
        payload = self._selected_payload(kwargs)
        self._emit_event('select', payload)
        return payload

    def _apply_filter_from_editor(self):
        self.filter(self.filter_edit.text())

    def _set_filter(self, text, update_text_field=True):
        self._filter_text = text or ''
        if update_text_field and self.filter_edit.text() != self._filter_text:
            self.filter_edit.setText(self._filter_text)
        if not self._filter_text:
            self._proxy.set_filter_predicate(None)
            self._filter_is_active = False
            return
        try:
            predicate, valid = _compile_filter_expr(self._filter_text, self._filterable_names)
        except Exception:
            predicate, valid = None, False
        self._proxy.set_filter_predicate(predicate if valid else None)
        self._filter_is_active = valid

    def _on_header_clicked(self, section):
        name = self.columns[section]
        current = self._current_sort
        sort_dir = 'asc'
        if current and current[0] == name and current[1] == 'asc':
            sort_dir = 'desc'
        self.sort_by(name, sort_dir)

    def _selection_anchor_row(self):
        visible = self._visible_ids()
        selected = self._selected_visible_ids()
        if not selected:
            return None
        return visible.index(selected[-1])

    def _on_row_clicked(self, index):
        row_id = self._visible_ids()[index.row()]
        mods = QApplication.keyboardModifiers()
        if mods & Qt.ControlModifier or mods & Qt.MetaModifier:
            self.select_toggle(row_id)
        elif mods & Qt.ShiftModifier:
            self.select_until(row_id)
        else:
            self.select([row_id])

    def get_selected_ids(self):
        visible = set(self._visible_ids())
        return [row_id for row_id in self._selected_ids if row_id in visible]

    def select_toggle(self, row_id):
        if row_id in self._selected_ids:
            self._selected_ids.remove(row_id)
        else:
            self._selected_ids.append(row_id)
        self._refresh_selection()
        return self._emit_selected()

    def select_until(self, row_id):
        visible = self._visible_ids()
        if row_id not in visible:
            return None
        anchor = self._selection_anchor_row()
        if anchor is None:
            return self.select([row_id])
        clicked = visible.index(row_id)
        imin, imax = sorted((anchor, clicked))
        for visible_id in visible[imin : imax + 1]:
            if visible_id not in self._selected_ids:
                self._selected_ids.append(visible_id)
        self._refresh_selection()
        return self._emit_selected()

    def get_sibling_id(self, row_id=None, direction='next'):
        selected = self.get_selected_ids()
        if row_id is None:
            row_id = selected[0] if selected else None
        if row_id is None:
            return None
        visible = self._visible_ids()
        if row_id not in visible:
            return None
        step = 1 if direction == 'next' else -1
        idx = visible.index(row_id) + step
        while 0 <= idx < len(visible):
            candidate = visible[idx]
            if not self._is_masked_id(candidate):
                return candidate
            idx += step
        return None

    def _move_to_sibling(self, row_id=None, direction='next'):
        if not self.get_selected_ids():
            return self._select_first_or_last('first')
        new_id = self.get_sibling_id(row_id, direction)
        if new_id is None:
            return None
        return self.select([new_id])

    def _select_first_or_last(self, which):
        visible = self._visible_ids()
        ordered = visible if which == 'first' else list(reversed(visible))
        for row_id in ordered:
            if not self._is_masked_id(row_id):
                return self.select([row_id])
        return None

    def sort_by(self, name, sort_dir='asc'):
        logger.log(5, 'Sort by `%s` %s.', name, sort_dir)
        if name not in self.columns:
            return
        column = self.columns.index(name)
        order = Qt.AscendingOrder if sort_dir == 'asc' else Qt.DescendingOrder
        self._current_sort = (name, sort_dir)
        self._proxy.sort(column, order)
        self._refresh_selection()
        self._fit_columns()
        if not self._no_emit:
            self._emit_event('table_sort', self._visible_ids())

    def filter(self, text=''):
        logger.log(5, 'Filter table with `%s`.', text)
        self._set_filter(text, update_text_field=True)
        self._refresh_selection()
        self._fit_columns()
        if self._filter_is_active and not self._no_emit:
            self._emit_event('table_filter', self._visible_ids())

    def _async_return(self, value, callback=None):
        self._schedule_callback(callback, value)
        return value

    def get_ids(self, callback=None):
        return self._async_return(self._visible_ids(), callback)

    def get_next_id(self, callback=None):
        return self._async_return(self.get_sibling_id(None, 'next'), callback)

    def get_previous_id(self, callback=None):
        return self._async_return(self.get_sibling_id(None, 'previous'), callback)

    def first(self, callback=None):
        return self._async_return(self._select_first_or_last('first'), callback)

    def last(self, callback=None):
        return self._async_return(self._select_first_or_last('last'), callback)

    def next(self, callback=None):
        return self._async_return(self._move_to_sibling(None, 'next'), callback)

    def previous(self, callback=None):
        return self._async_return(self._move_to_sibling(None, 'previous'), callback)

    def select(self, ids, callback=None, **kwargs):
        ids = _uniq(ids)
        assert all(_is_integer(_) for _ in ids)
        visible = set(self._visible_ids())
        self._selected_ids = [row_id for row_id in ids if row_id in visible]
        self._refresh_selection()
        payload = self._emit_selected(kwargs)
        return self._async_return(payload, callback)

    def scroll_to(self, id):
        index = self._proxy_index_for_id(id)
        if index.isValid():
            self.table_view.scrollTo(index)

    def set_busy(self, busy):
        self.debouncer.isBusy = bool(busy)

    def get(self, id, callback=None):
        row = self._model.row_by_id(id)
        out = dict(row) if row is not None else None
        return self._async_return(out, callback)

    def _ensure_list(self, objects):
        if isinstance(objects, dict):
            return [objects]
        return list(objects)

    def add(self, objects):
        objects = self._ensure_list(objects)
        if not objects:
            return
        data = self._model._rows + objects
        self._model.set_rows(data)
        if self._current_sort:
            self._no_emit = True
            self.sort_by(*self._current_sort)
            self._no_emit = False
        self._refresh_selection()
        self._fit_columns()

    def change(self, objects):
        objects = self._ensure_list(objects)
        if not objects:
            return
        updated = {obj['id']: obj for obj in objects}
        changed = False
        for row in self._model._rows:
            patch = updated.get(row['id'])
            if patch:
                row.update(patch)
                changed = True
        if not changed:
            return
        if self._model.rowCount() and self._model.columnCount():
            top_left = self._model.index(0, 0)
            bottom_right = self._model.index(
                self._model.rowCount() - 1, self._model.columnCount() - 1
            )
            self._model.dataChanged.emit(top_left, bottom_right)
        if self._current_sort:
            self._proxy.sort(
                self.columns.index(self._current_sort[0]),
                Qt.AscendingOrder if self._current_sort[1] == 'asc' else Qt.DescendingOrder,
            )
        self._refresh_selection()
        self._fit_columns()

    def remove(self, ids):
        ids = set(ids)
        if not ids:
            return
        self._selected_ids = [row_id for row_id in self._selected_ids if row_id not in ids]
        self._model.set_rows([row for row in self._model._rows if row['id'] not in ids])
        self._refresh_selection()
        self._fit_columns()

    def remove_all(self):
        self._selected_ids = []
        self._model.set_rows([])
        self._refresh_selection()
        self._fit_columns()

    def remove_all_and_add(self, objects):
        objects = self._ensure_list(objects)
        if not objects:
            return self.remove_all()
        self._selected_ids = []
        self._model.set_rows(objects)
        if self._current_sort:
            self._no_emit = True
            self.sort_by(*self._current_sort)
            self._no_emit = False
        self._refresh_selection()
        self._fit_columns()

    def get_selected(self, callback=None):
        return self._async_return(self.get_selected_ids(), callback)

    def get_current_sort(self, callback=None):
        value = list(self._current_sort) if self._current_sort else None
        return self._async_return(value, callback)

    def set_selected_index_offset(self, n):
        self._selected_index_offset = n
        self.table_view.viewport().update()

    def clear_temporary_files(self):
        """Compatibility no-op kept for callers from the removed WebEngine path."""
        return

    def sizeHint(self):
        return QSize(400, 400)

    def minimumSizeHint(self):
        return QSize(150, 150)

    def eval_js(self, expr, callback=None):
        expr = expr.strip()
        result = None
        emit_match = re.fullmatch(r'table\.emit\("([^"]+)",\s*(.+)\);?', expr)
        if emit_match:
            event_name, raw_value = emit_match.groups()
            try:
                result = json.loads(raw_value)
            except Exception:
                result = raw_value
            self._emit_event(event_name, result)
        elif expr == 'table.debouncer.isBusy':
            result = self.debouncer.isBusy
        elif expr.startswith('table._setSelectedIndexOffset('):
            number = int(re.search(r'\((\d+)\)', expr).group(1))
            self.set_selected_index_offset(number)
        elif expr == 'table.selected()':
            result = self.get_selected_ids()
        elif expr == 'table._getIds();':
            result = self._visible_ids()
        elif expr == 'table._currentSort()':
            result = list(self._current_sort) if self._current_sort else None
        else:
            logger.warning('Unsupported eval_js expression in native table: %s', expr)
        self._schedule_callback(callback, result)
        return result


# -----------------------------------------------------------------------------
# KeyValueWidget
# -----------------------------------------------------------------------------


class KeyValueWidget(QWidget):
    """A Qt widget that displays a simple form where each field has a name, a type, and accept
    user input."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._items = []
        self._layout = QGridLayout(self)

    def add_pair(self, name, default=None, vtype=None):
        """Add a key-value pair.

        Parameters
        ----------

        name : str
        default : object
        vtype : str
            Can be 'str' (text box), 'int' (spin box), 'float' (spin box), 'bool' (checkbox),
            'mutiline' (text edit for multiline str), or 'list' (several widgets).
        """
        if isinstance(default, list):
            # Take lists into account.
            for i, value in enumerate(default):
                self.add_pair(f'{name}[{i}]', default=value, vtype=vtype)
            return
        if vtype is None and default is not None:
            vtype = type(default).__name__
        if vtype == 'str':
            widget = QLineEdit(self)
            widget.setText(default or '')
        elif vtype == 'multiline':
            widget = QPlainTextEdit(self)
            widget.setPlainText(default or '')
            widget.setMinimumHeight(200)
            widget.setMaximumHeight(400)
        elif vtype == 'int':
            widget = QSpinBox(self)
            widget.setMinimum(-(10**9))
            widget.setMaximum(10**9)
            widget.setValue(default or 0)
        elif vtype == 'float':
            widget = QDoubleSpinBox(self)
            widget.setMinimum(-1e9)
            widget.setMaximum(+1e9)
            widget.setValue(default or 0)
        elif vtype == 'bool':
            widget = QCheckBox(self)
            widget.setChecked(default is True)
        else:  # pragma: no cover
            raise ValueError(f'Not supported vtype: {vtype}.')

        widget.setMaximumWidth(400)

        label = QLabel(name, self)
        label.setMaximumWidth(150)

        row = len(self._items)
        self._layout.addWidget(label, row, 0)
        self._layout.addWidget(widget, row, 1)
        self.setLayout(self._layout)
        self._items.append((name, vtype, default, widget))

    @property
    def names(self):
        """List of field names."""
        return sorted({i[0] if '[' not in i[0] else i[0][: i[0].index('[')] for i in self._items})

    def get_widget(self, name):
        """Get the widget of a field."""
        for name_, vtype, default, widget in self._items:
            if name == name_:
                return widget

    def get_value(self, name):
        """Get the default or user-entered value of a field."""
        # Detect if the requested name is a list type.
        names = {i[0] for i in self._items}
        if f'{name}[0]' in names:
            out = []
            i = 0
            namei = f'{name}[{i}]'
            while namei in names:
                out.append(self.get_value(namei))
                i += 1
                namei = f'{name}[{i}]'
            return out
        for name_, vtype, default, widget in self._items:
            if name_ == name:
                if vtype == 'str':
                    return str(widget.text())
                elif vtype == 'multiline':
                    return str(widget.toPlainText())
                elif vtype == 'int':
                    return int(widget.text())
                elif vtype == 'float':
                    return float(widget.text().replace(',', '.'))
                elif vtype == 'bool':
                    return bool(widget.isChecked())

    def attach(self, gui):  # pragma: no cover
        """Add the view to a GUI."""
        gui.add_view(self)

    def to_dict(self):
        """Return the key-value mapping dictionary as specified by the user inputs and defaults."""
        return {name: self.get_value(name) for name in self.names}
