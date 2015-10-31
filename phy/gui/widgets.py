# -*- coding: utf-8 -*-

"""HTML widgets for GUIs."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import logging

from .qt import QWebView

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Table
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
    title = 'Widget'
    body = ''

    def __init__(self):
        super(HTMLWidget, self).__init__()
        self._styles = [_DEFAULT_STYLES]
        self._header = ''
        self._body = ''

    def html(self):
        return self.page().mainFrame().toHtml()

    def add_styles(self, s):
        self._styles.append(s)

    def add_header(self, h):
        self._header += (h + '\n')

    def set_body(self, s):
        self._body = s

    def build(self):
        styles = '\n\n'.join(self._styles)
        html = _PAGE_TEMPLATE.format(title=self.title,
                                     styles=styles,
                                     header=self._header,
                                     body=self._body,
                                     )
        self.setHtml(html)

    def show(self):
        # Build if no HTML has been set.
        if self.html() == '<html><head></head><body></body></html>':
            self.build()
        return super(HTMLWidget, self).show()
