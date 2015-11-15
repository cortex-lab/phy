# -*- coding: utf-8 -*-

"""HTML widgets for GUIs."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import json
import logging
import os.path as op

from six import text_type

from .qt import (QWebView, QWebPage, QUrl, QWebSettings, QVariant,
                 pyqtSlot,
                 _wait_signal,
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
        font-size: 14pt;
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
        logger.debug(msg)  # pragma: no cover


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

    def _build(self):
        """Build the full HTML source."""
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
        return out.toPyObject() if isinstance(out, QVariant) else out

    @pyqtSlot(str, str)
    def _emit_from_js(self, name, arg_json):
        self.emit(text_type(name), json.loads(text_type(arg_json)))

    def show(self):
        with _wait_signal(self.loadFinished, 20):
            self._build()
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
        self._columns = [('id', (lambda _: _), {})]

    def add_column(self, func, name=None, options=None):
        """Add a column function which takes an id as argument and
        returns a value."""
        assert func
        name = name or func.__name__
        options = options or {}
        self._columns.append([name, func, options])
        return func

    def get_column(self, name):
        for col in self._columns:
            if col[0] == name:
                return col

    @property
    def column_names(self):
        return [name for (name, func, options) in self._columns
                if options.get('show', True)]

    def _get_row(self, id):
        """Create a row dictionary for a given object id."""
        return {name: func(id) for (name, func, options) in self._columns}

    def set_rows(self, ids):
        """Set the rows of the table."""
        items = [self._get_row(id) for id in ids]
        data = _create_json_dict(items=items,
                                 cols=self.column_names,
                                 )
        self.eval_js('table.setData({});'.format(data))

    def sort_by(self, header, dir='asc'):
        """Sort by a given variable."""
        self.eval_js('table.sortBy("{}", "{}");'.format(header, dir))

    def next(self):
        """Select the next non-skip row."""
        self.eval_js('table.next();')

    def previous(self):
        """Select the previous non-skip row."""
        self.eval_js('table.previous();')

    def select(self, ids, do_emit=True):
        """Select some rows."""
        do_emit = str(do_emit).lower()
        self.eval_js('table.select({}, {});'.format(dumps(ids), do_emit))

    @property
    def selected(self):
        """Currently selected rows."""
        return [int(_) for _ in self.eval_js('table.selected')]

    @property
    def current_sort(self):
        return tuple(self.eval_js('table.currentSort()'))
