# -*- coding: utf-8 -*-

"""HTML widgets for GUIs."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging
import os.path as op

from .qt import QWebView, QUrl, QWebSettings, pyqtSlot

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
        self._styles = [_DEFAULT_STYLES]
        self._header = ''
        self._body = ''
        self.add_to_js('widget', self)

    def add_to_js(self, name, var):
        """Add an object to Javascript."""
        frame = self.page().mainFrame()
        frame.addToJavaScriptWindowObject(name, var)

    def eval_js(self, expr):
        """Evaluate a Javascript expression."""
        frame = self.page().mainFrame()
        frame.evaluateJavaScript(expr)

    def html(self):
        return self.page().mainFrame().toHtml()

    def add_styles(self, s):
        self._styles.append(s)

    def add_script_src(self, filename):
        self.add_header('<script src="{}"></script>'.format(filename))

    def add_header(self, h):
        self._header += (h + '\n')

    @pyqtSlot(str)
    def set_body(self, s):
        self._body = s

    def build(self):
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

    def show(self):
        # Build if no HTML has been set.
        if self.html() == '<html><head></head><body></body></html>':
            self.build()
        return super(HTMLWidget, self).show()


# -----------------------------------------------------------------------------
# HTML table
# -----------------------------------------------------------------------------

_TABLE_STYLE = r"""

th.sort-header::-moz-selection { background:transparent; }
th.sort-header::selection  { background:transparent; }
th.sort-header { cursor:pointer; }

table th.sort-header:after {
    content: "\25B2";
    float: right;
    margin-left: 3px;
    margin-right: 15px;
    visibility: hidden;
}

table th.sort-header:hover:after {
    visibility: visible;
}

table th.sort-up:after {
    content: "\25BC";
}
table th.sort-down:after {
    content: "\25B2";
}

table th.sort-up:after,
table th.sort-down:after,
table th.sort-down:hover:after {
    visibility: visible;
}

"""


class Table(HTMLWidget):
    def __init__(self):
        super(Table, self).__init__()
        self.add_styles(_TABLE_STYLE)
        self.add_script_src('tablesort.min.js')
