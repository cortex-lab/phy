# -*- coding: utf-8 -*-

"""HTML widgets for GUIs."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import json
import logging
from functools import partial

from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager

from .qt import (
    QObject, QWidget, QGridLayout, QPlainTextEdit, QTableWidget, QTableWidgetItem,
    QLabel, QLineEdit, QCheckBox, QSpinBox, QDoubleSpinBox, Qt, QAbstractItemView,
    QEvent, qApp, QVBoxLayout, QColor, pyqtSlot, _static_abs_path, _block, Debouncer)
from phylib.utils import emit, connect
from phy.utils.color import colormaps, _is_bright
from phylib.utils._misc import _CustomEncoder, read_text, _pretty_floats
from phylib.utils._types import _is_integer
from phylib.io.array import _flatten

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# IPython widget
# -----------------------------------------------------------------------------

class IPythonView(RichJupyterWidget):
    """A view with an IPython console living in the same Python process as the GUI."""

    def __init__(self, *args, **kwargs):
        super(IPythonView, self).__init__(*args, **kwargs)
        title = "IPython widget"
        self.setWindowTitle(title)
        self.setObjectName(title)

    def start_kernel(self):
        """Start the IPython kernel."""

        logger.debug("Starting the kernel.")

        self.kernel_manager = QtInProcessKernelManager()
        self.kernel_manager.start_kernel(show_banner=False)
        self.kernel_manager.kernel.gui = 'qt'
        self.kernel = self.kernel_manager.kernel
        self.shell = self.kernel.shell

        try:
            self.kernel_client = self.kernel_manager.client()
            self.kernel_client.start_channels()
        except Exception as e:  # pragma: no cover
            logger.error("Could not start IPython kernel: %s.", str(e))

        self.set_default_style('linux')
        self.exit_requested.connect(self.stop)

    def inject(self, **kwargs):
        """Inject variables into the IPython namespace."""
        logger.debug("Injecting variables into the kernel: %s.", ', '.join(kwargs.keys()))
        try:
            self.kernel.shell.push(kwargs)
        except Exception as e:  # pragma: no cover
            logger.error("Could not inject variables to the IPython kernel: %s.", str(e))

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
        logger.debug("Stopping the kernel.")
        try:
            self.kernel_client.stop_channels()
            self.kernel_manager.shutdown_kernel()
        except Exception as e:  # pragma: no cover
            logger.error("Could not stop the IPython kernel: %s.", str(e))


def _uniq(seq):
    """Return the list of unique integers in a sequence, by keeping the order."""
    seen = set()
    seen_add = seen.add
    return [int(x) for x in seq if not (x in seen or seen_add(x))]


# -----------------------------------------------------------------------------
# Table
# -----------------------------------------------------------------------------

def dumps(o):
    """Dump a JSON object into a string, with pretty floats."""
    return json.dumps(_pretty_floats(o), cls=_CustomEncoder)


def _color_styles():
    """Use colormap colors in table widget."""
    return '\n'.join(
        '''
        #table .color-%d > td[class='id'] {
            background-color: rgb(%d, %d, %d);
            %s
        }
        ''' % (i, r, g, b, 'color: #000 !important;' if _is_bright((r, g, b)) else '')
        for i, (r, g, b) in enumerate(colormaps.default * 255))


class FilterForm(QLineEdit):
    pass


FILTER_PLACEHOLDER = "filter string, e.g. `count > 1e6`"


class Table(QTableWidget):
    """A sortable table with support for selection."""

    _data = None  # keep a dictionary mapping id to a dictionary {column: value}
    _mask_name = None  # name of the boolean field indicating whether an item is masked
    _sort = None
    _sel = ()
    _do_raise_select = True

    def __init__(self, *args, columns=None, data=None, sort=None, title='', mask_name='is_masked'):
        super(QTableWidget, self).__init__(0, 0, *args)

        self._data = {}
        self._mask_name = mask_name
        self._init_table(columns=columns, data=data, sort=sort)

        # Layout with the filter box and the table.
        layout = QVBoxLayout(*args)

        # Filter box.
        self.filter_form = FilterForm(*args)
        self.filter_form.setClearButtonEnabled(True)
        self.filter_form.setPlaceholderText(FILTER_PLACEHOLDER)
        self.filter_form.textChanged.connect(self.filter)
        self.filter_form.editingFinished.connect(self.filter)

        layout.addWidget(self.filter_form)
        layout.addWidget(self)

        self.container = QWidget()
        self.container.setLayout(layout)
        self.container.setWindowTitle('Table')

        self._init_events()

    def _init_table(self, columns=None, data=None, sort=None):
        """Build the table."""

        columns = columns or ['id']
        assert 'id' in columns
        assert columns.index('id') == 0  # HACK: used when converting row_idx <=> id
        # columns.remove('id')  # NOTE: keep the id column to enable sort by id

        data = data or []

        self.columns = columns

        emit('pre_build', self)

        # Fill in the table.
        self.clearContents()
        n_cols = len(columns)
        n_rows = len(data)

        # Set table size.
        self.setColumnCount(n_cols)

        # Set column names.
        self.setHorizontalHeaderLabels(columns)

        # Hide vertical header.
        # self.verticalHeader().setVisible(False)

        # Set the rows.
        self.add(data)

    def _init_events(self):
        connect(event='select', sender=self, func=lambda *args: self.update(), last=True)
        connect(event='ready', sender=self, func=lambda *args: self._set_ready())

        # Event when the sorting changes.
        header = self.horizontalHeader()

        @header.sortIndicatorChanged.connect
        def sort_changed(col_idx, order):
            emit('table_sort', self, self.get_ids())

        @self.itemSelectionChanged.connect
        def selection_changed():
            # Flag that enables or disables the emission of the select event in reaction to
            # cluster selection via clicking.
            if self._do_raise_select:
                self._emit_select()

    # Overriden methods
    # ---------------------------------------------------------------------------------------------

    def show(self):
        super(Table, self).show()
        self.container.show()

    def close(self):
        super(Table, self).close()
        self.container.close()

    def resize(self, *args):
        self.container.resize(*args)

    # Internal util functions
    # ---------------------------------------------------------------------------------------------

    def _emit_select(self):
        """Emit a select event."""
        sel = self.get_selected()

        # HACK: ensure that the select event is only emitted if the selection really changes.
        # The reason is that this Qt slot may be called when the selected column changes but
        # not the row.
        if tuple(sel) == self._sel:
            return
        self._sel = tuple(sel)
        sel_next = self.get_next_id(sel[0] if sel else None)
        emit('select', self, {'selected': sel, 'next': sel_next})

    def _get_value(self, id, col_name):
        """Return the value of an item."""
        return self._data.get(id, {}).get(col_name, None)

    def _set_value(self, id, col_name, value):
        """Set the value of an item."""
        self._data.get(id, {})[col_name] = value

        # Find the row and column index of the corresponding item.
        row_idx = self._id2row(id)
        assert col_name in self.columns
        col_idx = self.columns.index(col_name)
        self._set_item_value(row_idx, col_idx, value)

    def _row2id(self, row_idx):
        assert row_idx is not None
        assert row_idx >= 0
        item = self.item(row_idx, 0)
        if not item:
            raise ValueError(f"Row {row_idx} not found.")
            return -1
        id = item.data(Qt.DisplayRole)
        return int(id) if id is not None else -1

    def _id2row(self, id):
        assert id is not None
        assert id >= 0
        for row_idx in range(self.rowCount()):
            if self._row2id(row_idx) == id:
                return row_idx
        raise ValueError(f"Item with id {id} not found in the table.")

    def _row_items(self, row_idx):
        assert row_idx is not None
        assert row_idx >= 0
        return [self.item(row_idx, col_idx) for col_idx in range(self.columnCount())]

    def _hide_row(self, row_idx):
        self.hideRow(row_idx)

    def _show_row(self, row_idx):
        self.showRow(row_idx)

    def _is_masked(self, id):
        """Return whether an item is masked or not."""
        return self._data.get(id, {}).get(self._mask_name, False)

    def get(self, id):
        """Get the object given its id."""
        return self._data.get(id, {})

    # Selection
    # ---------------------------------------------------------------------------------------------

    def get_selected(self):
        """Get the currently selected rows."""
        return _uniq([self._row2id(item.row()) for item in self.selectedItems()])

    def select(self, ids, **kwargs):
        """Select some rows in the table from Python."""
        ids = _uniq(ids)
        assert all(_is_integer(_) for _ in ids)
        rows = [self._id2row(id) for id in ids]
        items = _flatten([self._row_items(row) for row in rows])

        # HACK: avoid raising the select event automatically when changing the selection status of
        # the items. Raise it manually instead.
        do_raise_select = self._do_raise_select
        self._do_raise_select = False
        self.clearSelection()
        for item in items:
            item.setSelected(True)
        self._do_raise_select = do_raise_select
        self._emit_select()

    # Scrolling
    # ---------------------------------------------------------------------------------------------

    def scroll_to(self, id):
        """Scroll until a given row is visible."""
        row_idx = self._id2row(id)
        items = self._row_items(row_idx)
        assert items
        self.scrollToItem(items[0], QAbstractItemView.PositionAtCenter)

    # Wizard
    # ---------------------------------------------------------------------------------------------

    def get_next_id(self, id=None):
        """Return the next unmasked id after the specified id."""
        row_idx = self._id2row(id) if id is not None else -1
        # Go through all items after the specified id.
        for row_idx in range(row_idx + 1, self.rowCount()):
            nid = self._row2id(row_idx)
            if not self._is_masked(nid):
                return nid
        return None

    def get_previous_id(self, id=None):
        """Return the previous unmasked id before the specified id."""
        row_idx = self._id2row(id) if id is not None else self.rowCount() - 1
        # Go through all items after the specified id.
        for row_idx in range(row_idx - 1, -1, -1):
            nid = self._row2id(row_idx)
            if not self._is_masked(nid):
                return nid
        return None

    def first(self):
        """Select the first item."""
        self.select([self.get_next_id()])

    def last(self):
        """Select the last item."""
        self.select([self.get_previous_id()])

    def next(self):
        """Select the next non-skipped row."""
        sel = self.get_selected()
        if not sel:
            return self.first()
        # First selected item.
        id = sel[0]
        # Find the next unmasked id.
        nid = self.get_next_id(id)
        # Select it.
        if nid is not None:
            self.select([nid])

    def previous(self):
        """Select the previous non-skipped row."""
        sel = self.get_selected()
        if not sel:
            return self.last()
        # First selected item.
        id = sel[0]
        # Find the previous unmasked id.
        nid = self.get_previous_id(id)
        # Select it.
        if nid is not None:
            self.select([nid])

    def get_ids(self):
        """Get the list of ids."""
        return [self._row2id(row_idx) for row_idx in range(self.rowCount())]

    # Sorting
    # ---------------------------------------------------------------------------------------------

    def sort_by(self, name, sort_dir='asc'):
        """Sort by a given variable."""
        logger.log(5, "Sort by `%s` %s.", name, sort_dir)
        self._sort = (name, sort_dir)
        assert name in self.columns
        col_idx = self.columns.index(name)
        order = Qt.AscendingOrder if sort_dir == 'asc' else Qt.DescendingOrder
        self.sortItems(col_idx, order)

    def get_current_sort(self):
        """Get the current sort as a tuple `(name, dir)`."""
        return self._sort

    # Filtering
    # ---------------------------------------------------------------------------------------------

    def filter(self, text=''):
        """Filter the view with a Python expression."""
        text0 = text
        for col in self.columns:
            text = text.replace(col, f'row_dict.get("{col}", None)')
        # logger.log(10, "Filter table with `%s`.", text)

        # All ids.
        ids = self.get_ids()

        # Compile the filter function.
        try:
            f = eval('lambda row_dict: ' + text)
            f(self._data[ids[0]])
        except Exception as e:
            logger.log(5, f"Filter `{text0}` is invalid.")
            text = ''

        # Ids to keep.
        kept = [id for id in ids if not text or f(self._data[id])]

        # Emit an event.
        emit('table_filter', self, kept)

        # Ids to filter out.
        to_hide = set(ids) - set(kept)

        # Hide rows.
        for id in to_hide:
            self._hide_row(self._id2row(id))

        # Show the others.
        for id in kept:
            self._show_row(self._id2row(id))

    def shown_ids(self):
        """Return the list of filtered ids."""
        return [
            self._row2id(row_idx) for row_idx in range(self.rowCount())
            if not self.isRowHidden(row_idx)]

    # Update functions
    # ---------------------------------------------------------------------------------------------

    def _create_item(self, row_idx, col_idx, flags=Qt.ItemIsSelectable | Qt.ItemIsEnabled):
        item = QTableWidgetItem()
        item.setFlags(flags)
        self.setItem(row_idx, col_idx, item)

    def _set_item_value(self, row_idx, col_idx, value):
        item = self.item(row_idx, col_idx)
        assert item
        item.setData(Qt.EditRole, value)
        item.setData(Qt.DisplayRole, str(value))

    def _set_item_style(self, row_idx, col_idx, d):
        item = self.item(row_idx, col_idx)
        assert item
        bg = d.get('_background', None)
        fg = d.get('_foreground', None)
        if bg:
            item.setBackground(QColor(bg))
        if fg:
            item.setForeground(QColor(fg))

    def add(self, data):
        """Add objects to the table."""

        self.setSortingEnabled(False)

        data = data or []
        self._data.update({d['id']: d for d in data})

        flags = Qt.ItemIsSelectable | Qt.ItemIsEnabled

        # Previous row count.
        prev_n_rows = self.rowCount()

        # New row count.
        new_n_rows = prev_n_rows + len(data)
        self.setRowCount(new_n_rows)

        for row_rel_idx, row_dict in enumerate(data):
            row_idx = row_rel_idx + prev_n_rows

            # Set the row id.
            id = row_dict['id']
            assert id >= 0
            self.setVerticalHeaderItem(row_idx, QTableWidgetItem(id))

            # Set the values.
            for col_idx, col_name in enumerate(self.columns):
                # Create the QTableWidgetItem object.
                self._create_item(row_idx, col_idx)

                # Set the item style.
                self._set_item_style(row_idx, col_idx, row_dict)

                # Set the item's data.
                self._set_item_value(row_idx, col_idx, row_dict.get(col_name, ''))

        self.setSortingEnabled(True)
        self.resizeColumnsToContents()

    def change(self, objects):
        """Change some objects."""
        self.setSortingEnabled(False)

        for row_dict in objects:
            id = row_dict['id']
            for col_name, value in row_dict.items():

                self._data.get(id, {})[col_name] = value

                # Find the row and column index of the corresponding item.
                row_idx = self._id2row(id)
                assert col_name in self.columns
                col_idx = self.columns.index(col_name)

                # Set the item's value.
                self._set_item_value(row_idx, col_idx, value)

                # Set the item style.
                self._set_item_style(row_idx, col_idx, row_dict)

        self.setSortingEnabled(True)

    def remove(self, ids):
        """Remove some objects from their ids."""
        for id in ids:
            row_idx = self._id2row(id)
            self.removeRow(row_idx)

    def remove_all(self):
        """Remove all rows in the table."""
        self.clearContents()
        self.setRowCount(0)

    def remove_all_and_add(self, objects):
        """Remove all rows in the table and add new objects."""
        self.remove_all()
        self.add(objects)


# -----------------------------------------------------------------------------
# KeyValueWidget
# -----------------------------------------------------------------------------

class KeyValueWidget(QWidget):
    """A Qt widget that displays a simple form where each field has a name, a type, and accept
    user input."""

    def __init__(self, *args, **kwargs):
        super(KeyValueWidget, self).__init__(*args, **kwargs)
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
                self.add_pair('%s[%d]' % (name, i), default=value, vtype=vtype)
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
            widget.setMinimum(int(-1e9))
            widget.setMaximum(int(+1e9))
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
            raise ValueError("Not supported vtype: %s." % vtype)

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
        return sorted(
            set(i[0] if '[' not in i[0] else i[0][:i[0].index('[')] for i in self._items))

    def get_widget(self, name):
        """Get the widget of a field."""
        for name_, vtype, default, widget in self._items:
            if name == name_:
                return widget

    def get_value(self, name):
        """Get the default or user-entered value of a field."""
        # Detect if the requested name is a list type.
        names = set(i[0] for i in self._items)
        if '%s[0]' % name in names:
            out = []
            i = 0
            namei = '%s[%d]' % (name, i)
            while namei in names:
                out.append(self.get_value(namei))
                i += 1
                namei = '%s[%d]' % (name, i)
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
