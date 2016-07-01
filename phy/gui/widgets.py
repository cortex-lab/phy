# -*- coding: utf-8 -*-

"""HTML widgets for GUIs."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from collections import OrderedDict
import json
import logging
import os.path as op

from six import text_type

from .qt import (QWebView, QWebPage, QUrl, QWebSettings,
                 QVariant, QPyNullVariant, QString,
                 pyqtSlot, _wait_signal,
                 )
from phy.utils import EventEmitter
from phy.utils._misc import _CustomEncoder

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# HTML widget
# -----------------------------------------------------------------------------

_DEFAULT_STYLES = """
    html, body, table {
        background-color: black;
        color: white;
        font-family: sans-serif;
        font-size: 12pt;
        margin: 5px 10px;
    }
"""


_PAGE_TEMPLATE = """
<html>
<head>
    <title>{title:s}</title>
    <style>
    {styles:s}
    </style>
    {header:s}
</head>
<body>

{body:s}

</body>
</html>
"""


class WebPage(QWebPage):
    def javaScriptConsoleMessage(self, msg, line, source):
        logger.debug("[%d] %s", line, msg)  # pragma: no cover


def _to_py(obj):  # pragma: no cover
    if isinstance(obj, QVariant):
        return obj.toPyObject()
    elif QString and isinstance(obj, QString):
        return text_type(obj)
    elif isinstance(obj, QPyNullVariant):
        return None
    elif isinstance(obj, list):
        return [_to_py(_) for _ in obj]
    elif isinstance(obj, tuple):
        return tuple(_to_py(_) for _ in obj)
    else:
        return obj


class HTMLWidget(QWebView):
    """An HTML widget that is displayed with Qt.

    Python methods can be called from Javascript with `widget.the_method()`.
    They must be decorated with `pyqtSlot(str)` or similar, depending on
    the parameters.

    """
    title = 'Widget'
    body = ''

    def __init__(self):
        super(HTMLWidget, self).__init__()
        self.settings().setAttribute(
            QWebSettings.LocalContentCanAccessRemoteUrls, True)
        self.settings().setAttribute(
            QWebSettings.DeveloperExtrasEnabled, True)
        self.setPage(WebPage())
        self._obj = None
        self._styles = [_DEFAULT_STYLES]
        self._header = ''
        self._body = ''
        self.add_to_js('widget', self)
        self._event = EventEmitter()
        self.add_header('''<script>
                        var emit = function (name, arg) {
                            widget._emit_from_js(name, JSON.stringify(arg));
                        };
                        </script>''')
        self._pending_js_eval = []

    # Events
    # -------------------------------------------------------------------------

    def emit(self, *args, **kwargs):
        return self._event.emit(*args, **kwargs)

    def connect_(self, *args, **kwargs):
        self._event.connect(*args, **kwargs)

    def unconnect_(self, *args, **kwargs):
        self._event.unconnect(*args, **kwargs)

    # Headers
    # -------------------------------------------------------------------------

    def add_styles(self, s):
        """Add CSS styles."""
        self._styles.append(s)

    def add_style_src(self, filename):
        """Link a CSS file."""
        self.add_header(('<link rel="stylesheet" type="text/css" '
                         'href="{}" />').format(filename))

    def add_script_src(self, filename):
        """Link a JS script."""
        self.add_header('<script src="{}"></script>'.format(filename))

    def add_header(self, h):
        """Add HTML code to the header."""
        self._header += (h + '\n')

    # HTML methods
    # -------------------------------------------------------------------------

    def set_body(self, s):
        """Set the HTML body."""
        self._body = s

    def add_body(self, s):
        """Add HTML code to the body."""
        self._body += '\n' + s + '\n'

    def html(self):
        """Return the full HTML source of the widget."""
        return self.page().mainFrame().toHtml()

    def build(self):
        """Build the full HTML source."""
        if self.is_built():  # pragma: no cover
            return
        with _wait_signal(self.loadFinished, 20):
            styles = '\n\n'.join(self._styles)
            html = _PAGE_TEMPLATE.format(title=self.title,
                                         styles=styles,
                                         header=self._header,
                                         body=self._body,
                                         )
            logger.log(5, "Set HTML: %s", html)
            static_dir = op.join(op.realpath(op.dirname(__file__)), 'static/')
            base_url = QUrl().fromLocalFile(static_dir)
            self.setHtml(html, base_url)

    def is_built(self):
        return self.html() != '<html><head></head><body></body></html>'

    # Javascript methods
    # -------------------------------------------------------------------------

    def add_to_js(self, name, var):
        """Add an object to Javascript."""
        frame = self.page().mainFrame()
        frame.addToJavaScriptWindowObject(name, var)

    def eval_js(self, expr):
        """Evaluate a Javascript expression."""
        if not self.is_built():
            self._pending_js_eval.append(expr)
            return
        logger.log(5, "Evaluate Javascript: `%s`.", expr)
        out = self.page().mainFrame().evaluateJavaScript(expr)
        return _to_py(out)

    @pyqtSlot(str, str)
    def _emit_from_js(self, name, arg_json):
        self.emit(text_type(name), json.loads(text_type(arg_json)))

    def show(self):
        self.build()
        super(HTMLWidget, self).show()
        # Call the pending JS eval calls after the page has been built.
        assert self.is_built()
        for expr in self._pending_js_eval:
            self.eval_js(expr)
        self._pending_js_eval = []


# -----------------------------------------------------------------------------
# HTML table
# -----------------------------------------------------------------------------

def dumps(o):
    return json.dumps(o, cls=_CustomEncoder)


def _create_json_dict(**kwargs):
    d = {}
    # Remove None elements.
    for k, v in kwargs.items():
        if v is not None:
            d[k] = v
    # The custom encoder serves for NumPy scalars that are non
    # JSON-serializable (!!).
    return dumps(d)


class Table(HTMLWidget):
    """A sortable table with support for selection."""

    _table_id = 'the-table'

    def __init__(self):
        super(Table, self).__init__()
        self.add_style_src('table.css')
        self.add_script_src('tablesort.min.js')
        self.add_script_src('tablesort.number.js')
        self.add_script_src('table.js')
        self.set_body('<table id="{}" class="sort"></table>'.format(
                      self._table_id))
        self.add_body('''<script>
                      var table = new Table(document.getElementById("{}"));
                      </script>'''.format(self._table_id))
        self._columns = OrderedDict()
        self._default_sort = (None, None)
        self.add_column(lambda _: _, name='id')

    def add_column(self, func, name=None, show=True):
        """Add a column function which takes an id as argument and
        returns a value."""
        assert func
        name = name or func.__name__
        if name == '<lambda>':
            raise ValueError("Please provide a valid name for " + name)
        d = {'func': func,
             'show': show,
             }
        self._columns[name] = d

        # Update the headers in the widget.
        data = _create_json_dict(cols=self.column_names,
                                 )
        self.eval_js('table.setHeaders({});'.format(data))

        return func

    @property
    def column_names(self):
        """List of column names."""
        return [name for (name, d) in self._columns.items()
                if d.get('show', True)]

    def _get_row(self, id):
        """Create a row dictionary for a given object id."""
        return {name: d['func'](id) for (name, d) in self._columns.items()}

    def set_rows(self, ids):
        """Set the rows of the table."""
        # NOTE: make sure we have integers and not np.generic objects.
        assert all(isinstance(i, int) for i in ids)

        # Determine the sort column and dir to set after the rows.
        sort_col, sort_dir = self.current_sort
        default_sort_col, default_sort_dir = self.default_sort

        sort_col = sort_col or default_sort_col
        sort_dir = sort_dir or default_sort_dir or 'desc'

        # Set the rows.
        logger.log(5, "Set %d rows in the table.", len(ids))
        items = [self._get_row(id) for id in ids]
        # Sort the rows before passing them to the widget.
        # if sort_col:
        #     items = sorted(items, key=itemgetter(sort_col),
        #                    reverse=(sort_dir == 'desc'))
        data = _create_json_dict(items=items,
                                 cols=self.column_names,
                                 )
        self.eval_js('table.setData({});'.format(data))

        # Sort.
        if sort_col:
            self.sort_by(sort_col, sort_dir)

    def sort_by(self, name, sort_dir='asc'):
        """Sort by a given variable."""
        logger.log(5, "Sort by `%s` %s.", name, sort_dir)
        self.eval_js('table.sortBy("{}", "{}");'.format(name, sort_dir))

    def next(self):
        """Select the next non-skipped row."""
        self.eval_js('table.next();')

    def previous(self):
        """Select the previous non-skipped row."""
        self.eval_js('table.previous();')

    def select(self, ids, do_emit=True):
        """Select some rows."""
        do_emit = str(do_emit).lower()
        self.eval_js('table.select({}, {});'.format(dumps(ids), do_emit))

    @property
    def default_sort(self):
        """Default sort as a pair `(name, dir)`."""
        return self._default_sort

    def set_default_sort(self, name, sort_dir='desc'):
        """Set the default sort column."""
        self._default_sort = name, sort_dir

    @property
    def selected(self):
        """Currently selected rows."""
        return [int(_) for _ in self.eval_js('table.selected') or ()]

    @property
    def current_sort(self):
        """Current sort: a tuple `(name, dir)`."""
        return tuple(self.eval_js('table.currentSort()') or (None, None))
