# -*- coding: utf-8 -*-

"""HTML widgets for GUIs."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import json
import logging
import os.path as op

from .qt import QWebView, QWebPage, QUrl, QWebSettings, pyqtSlot

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# HTML widget
# -----------------------------------------------------------------------------

_DEFAULT_STYLES = """
    html, body, table {
        background-color: black;
        color: white;
        font-family: sans-serif;
        font-size: 18pt;
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

    @pyqtSlot(str)
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

    # Javascript methods
    # -------------------------------------------------------------------------

    def add_to_js(self, name, var):
        """Add an object to Javascript."""
        frame = self.page().mainFrame()
        frame.addToJavaScriptWindowObject(name, var)

    def eval_js(self, expr):
        """Evaluate a Javascript expression."""
        frame = self.page().mainFrame()
        frame.evaluateJavaScript(expr)

    @pyqtSlot(str)
    def _set_from_js(self, obj):
        """Called from Javascript to pass any object to Python through JSON."""
        self._obj = json.loads(str(obj))

    def get_js(self, expr):
        """Evaluate a Javascript expression and get a Python object.

        This uses JSON serialization/deserialization under the hood.

        """
        self.eval_js('widget._set_from_js(JSON.stringify({}));'.format(expr))
        obj = self._obj
        self._obj = None
        return obj

    def show(self):
        """Show the widget.

        A build is triggered if necessary.

        """
        # Build if no HTML has been set.
        if self.html() == '<html><head></head><body></body></html>':
            self.build()
        return super(HTMLWidget, self).show()


# -----------------------------------------------------------------------------
# HTML table
# -----------------------------------------------------------------------------

def _create_json_dict(**kwargs):
    d = {}
    for k, v in kwargs.items():
        if v is not None:
            d[k] = v
    return json.dumps(d)


class Table(HTMLWidget):
    """A sortable table with support for selection."""

    _table_id = 'the-table'

    def __init__(self):
        super(Table, self).__init__()
        self.add_style_src('table.css')
        self.add_script_src('tablesort.min.js')
        self.add_script_src('table.js')
        self.set_body('<table id="{}" class="sort"></table>'.format(
                      self._table_id))
        self.add_body('''<script>
                      var table = new Table(document.getElementById("{}"));
                      </script>'''.format(self._table_id))
        self.build()

    def set_data(self, items, cols):
        """Set the rows and cols of the table."""
        data = _create_json_dict(items=items,
                                 cols=cols,
                                 )
        self.eval_js('table.setData({});'.format(data))

    def set_state(self, selected=None, sort_col=None, sort_dir=None):
        """Set the state of the widget."""
        state = _create_json_dict(selected=[int(_) for _ in selected],
                                  sortCol=sort_col,
                                  sortDir=sort_dir,
                                  )
        self.eval_js('table.setState({});'.format(state))

    def next(self):
        """Select the next non-skip row."""
        self.eval_js('table.next();')

    def previous(self):
        """Select the previous non-skip row."""
        self.eval_js('table.previous();')

    def select(self, ids):
        """Select some rows."""
        self.set_state(selected=ids)

    @property
    def state(self):
        """Current state."""
        return self.get_js("table.getState()")

    @property
    def selected(self):
        """Currently selected rows."""
        return [int(_) for _ in self.state.get('selected', [])]
