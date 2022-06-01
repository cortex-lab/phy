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
    QObject, QWidget, QGridLayout, QPlainTextEdit,
    QLabel, QLineEdit, QCheckBox, QSpinBox, QDoubleSpinBox,
    pyqtSlot, _static_abs_path, _block, Debouncer)
from phylib.utils import emit, connect
from phy.utils.color import colormaps, _is_bright
from phylib.utils._misc import _CustomEncoder, read_text, _pretty_floats
from phylib.utils._types import _is_integer

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


class Barrier(object):
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


class Table(QWidget):
    # TODO
    """A sortable table with support for selection.

    This table uses the following Javascript implementation: https://github.com/kwikteam/tablejs
    This Javascript class builds upon ListJS: https://listjs.com/

    """

    _ready = False

    def __init__(
            self, *args, columns=None, value_names=None, data=None, sort=None, title='',
            debounce_events=()):
        super(Table, self).__init__(*args, title=title, debounce_events=debounce_events)
        self._init_table(columns=columns, value_names=value_names, data=data, sort=sort)

    def eval_js(self, expr, callback=None):
        """Evaluate a Javascript expression.

        The `table` Javascript variable can be used to interact with the underlying Javascript
        table.

        The table has sortable columns, a filter text box, support for single and multi selection
        of rows. Rows can be skippable (used for ignored clusters in phy).

        The table can raise Javascript events that are relayed to Python. Objects are
        transparently serialized and deserialized in JSON. Basic types (numbers, strings, lists)
        are transparently converted between Python and Javascript.

        Parameters
        ----------

        expr : str
            A Javascript expression.
        callback : function
            A Python function that is called once the Javascript expression has been
            evaluated. It takes as input the output of the Javascript expression.

        """
        # Avoid JS errors when the table is not yet fully loaded.
        expr = 'if (typeof table !== "undefined") ' + expr
        return super(Table, self).eval_js(expr, callback=callback)

    def _init_table(self, columns=None, value_names=None, data=None, sort=None):
        """Build the table."""

        columns = columns or ['id']
        value_names = value_names or columns
        data = data or []

        b = self.builder
        b.set_body_src('index.html')

        b.add_style(_color_styles())

        self.data = data
        self.columns = columns
        self.value_names = value_names

        emit('pre_build', self)

        data_json = dumps(self.data)
        columns_json = dumps(self.columns)
        value_names_json = dumps(self.value_names)
        sort_json = dumps(sort)

        b.body += '''
        <script>
            var data = %s;

            var options = {
              valueNames: %s,
              columns: %s,
              sort: %s,
            };

            var table = new Table('table', options, data);

        </script>
        ''' % (data_json, value_names_json, columns_json, sort_json)
        self.build(lambda html: emit('ready', self))

        connect(event='select', sender=self, func=lambda *args: self.update(), last=True)
        connect(event='ready', sender=self, func=lambda *args: self._set_ready())

    def _set_ready(self):
        """Set the widget as ready."""
        self._ready = True

    def is_ready(self):
        """Whether the widget has been fully loaded."""
        return self._ready

    def sort_by(self, name, sort_dir='asc'):
        """Sort by a given variable."""
        logger.log(5, "Sort by `%s` %s.", name, sort_dir)
        self.eval_js('table.sort_("{}", "{}");'.format(name, sort_dir))

    def filter(self, text=''):
        """Filter the view with a Javascript expression."""
        logger.log(5, "Filter table with `%s`.", text)
        self.eval_js('table.filter_("{}", true);'.format(text))

    def get_ids(self, callback=None):
        """Get the list of ids."""
        self.eval_js('table._getIds();', callback=callback)

    def get_next_id(self, callback=None):
        """Get the next non-skipped row id."""
        self.eval_js('table.getSiblingId(undefined, "next");', callback=callback)

    def get_previous_id(self, callback=None):
        """Get the previous non-skipped row id."""
        self.eval_js('table.getSiblingId(undefined, "previous");', callback=callback)

    def first(self, callback=None):
        """Select the first item."""
        self.eval_js('table.selectFirst();', callback=callback)

    def last(self, callback=None):
        """Select the last item."""
        self.eval_js('table.selectLast();', callback=callback)

    def next(self, callback=None):
        """Select the next non-skipped row."""
        self.eval_js('table.moveToSibling(undefined, "next");', callback=callback)

    def previous(self, callback=None):
        """Select the previous non-skipped row."""
        self.eval_js('table.moveToSibling(undefined, "previous");', callback=callback)

    def select(self, ids, callback=None, **kwargs):
        """Select some rows in the table from Python.

        This function calls `table.select()` in Javascript, which raises a Javascript event
        relayed to Python. This sequence of actions is the same when the user selects
        rows directly in the HTML view.

        """
        ids = _uniq(ids)
        assert all(_is_integer(_) for _ in ids)
        self.eval_js('table.select({}, {});'.format(dumps(ids), dumps(kwargs)), callback=callback)

    def scroll_to(self, id):
        """Scroll until a given row is visible."""
        self.eval_js('table._scrollTo({});'.format(id))

    def set_busy(self, busy):
        """Set the busy state of the GUI."""
        self.eval_js('table.setBusy({});'.format('true' if busy else 'false'))

    def get(self, id, callback=None):
        """Get the object given its id."""
        self.eval_js('table.get("id", {})[0]["_values"]'.format(id), callback=callback)

    def add(self, objects):
        """Add objects object to the table."""
        if not objects:
            return
        self.eval_js('table.add_({});'.format(dumps(objects)))

    def change(self, objects):
        """Change some objects."""
        if not objects:
            return
        self.eval_js('table.change_({});'.format(dumps(objects)))

    def remove(self, ids):
        """Remove some objects from their ids."""
        if not ids:
            return
        self.eval_js('table.remove_({});'.format(dumps(ids)))

    def remove_all(self):
        """Remove all rows in the table."""
        self.eval_js('table.removeAll();')

    def remove_all_and_add(self, objects):
        """Remove all rows in the table and add new objects."""
        if not objects:
            return self.remove_all()
        self.eval_js('table.removeAllAndAdd({});'.format(dumps(objects)))

    def get_selected(self, callback=None):
        """Get the currently selected rows."""
        self.eval_js('table.selected()', callback=callback)

    def get_current_sort(self, callback=None):
        """Get the current sort as a tuple `(name, dir)`."""
        self.eval_js('table._currentSort()', callback=callback)


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
