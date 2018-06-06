# -*- coding: utf-8 -*-

"""HTML widgets for GUIs."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import json
import logging

from six import text_type

from .qt import WebView, QWebChannel, pyqtSlot, block, _abs_path
from phy.utils import EventEmitter
from phy.utils._misc import _CustomEncoder, _read_text

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# HTML widget
# -----------------------------------------------------------------------------

_DEFAULT_STYLE = """
    html, body, table {
        background-color: black;
        color: white;
        font-family: sans-serif;
        font-size: 12pt;
        margin: 5px 10px;
    }
"""


_DEFAULT_SCRIPT = """
    function onWidgetReady(callback) {
        document.addEventListener("DOMContentLoaded", function() {
            new QWebChannel(qt.webChannelTransport, function(channel) {
                var widget = channel.objects.widget;

                widget.emitPy = function(name, arg) {
                    widget._emit_from_js(name, JSON.stringify(arg));
                };
                window.emit = widget.emitPy;
                window.widget = widget;

                callback(widget);
            });
        });
    };
"""


_PAGE_TEMPLATE = """
<html>
<head>
    <title>{title:s}</title>
    {header:s}
</head>
<body>

{body:s}

</body>
</html>
"""


def _uniq(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


class HTMLBuilder(object):
    def __init__(self, title=''):
        self.title = title
        self.headers = []
        self.body = ''
        self.add_style(_DEFAULT_STYLE)

    def add_style(self, s):
        self.add_header('<style>\n{}\n</style>'.format(s))

    def add_style_src(self, filename):
        self.add_header(('<link rel="stylesheet" type="text/css" '
                         'href="{}" />').format(filename))

    def add_script(self, s):
        self.add_header('<script>{}</script>'.format(s))

    def add_script_src(self, filename):
        self.add_header('<script src="{}"></script>'.format(filename))

    def add_header(self, s):
        self.headers.append(s)

    def set_body_src(self, filename):
        path = _abs_path(filename)
        self.set_body(_read_text(path))

    def set_body(self, body):
        self.body = body

    def _build_html(self):
        header = '\n'.join(self.headers)
        html = _PAGE_TEMPLATE.format(title=self.title,
                                     header=header,
                                     body=self.body,
                                     )
        return html

    @property
    def html(self):
        return self._build_html()


class HTMLWidget(WebView):
    """An HTML widget that is displayed with Qt."""
    def __init__(self, title=''):
        super(HTMLWidget, self).__init__()
        self._event = EventEmitter()

        self.channel = QWebChannel(self.page())
        self.page().setWebChannel(self.channel)
        self.channel.registerObject('widget', self)

        self.builder = HTMLBuilder(title=title)
        self.builder.add_script_src('qrc:///qtwebchannel/qwebchannel.js')
        self.builder.add_script(_DEFAULT_SCRIPT)

    def build(self):
        self.set_html_sync(self.builder.html)

    def block_until_loaded(self):
        block(lambda: self.eval_js("typeof(window.widget) !== 'undefined'"))

    # Events
    # -------------------------------------------------------------------------

    def emit(self, *args, **kwargs):
        return self._event.emit(*args, **kwargs)

    def connect_(self, *args, **kwargs):
        self._event.connect(*args, **kwargs)

    def unconnect_(self, *args, **kwargs):
        self._event.unconnect(*args, **kwargs)

    # Javascript methods
    # -------------------------------------------------------------------------

    def eval_js(self, expr, callback=None, sync=True):
        """Evaluate a Javascript expression."""
        if not sync:
            return self.page().runJavaScript(expr, callback or (lambda _: _))
        self._js_done = False
        self._js_result = None

        assert not callback

        def callback(res):
            self._js_done = True
            self._js_result = res

        self.page().runJavaScript(expr, callback)

        # Synchronous execution.
        block(lambda: self._js_done)

        res = self._js_result
        self._js_done = False
        self._js_result = None

        return res

    @pyqtSlot(str, str)
    def _emit_from_js(self, name, arg_json):
        self.emit(text_type(name), json.loads(text_type(arg_json)))


# -----------------------------------------------------------------------------
# HTML table
# -----------------------------------------------------------------------------

def dumps(o):
    return json.dumps(o, cls=_CustomEncoder)


class Table(HTMLWidget):
    """A sortable table with support for selection."""

    def __init__(self, columns=None, data=None, title=''):
        super(Table, self).__init__(title=title)
        self.columns = columns or []
        b = self.builder
        b.set_body_src('index.html')
        data_json = dumps(data)
        columns_json = dumps(columns)
        b.body += '''
        <script>
            var data = %s;

            var options = {
              valueNames: %s.concat(["_meta"]),
              columns: %s,
            };

            var table = new Table('table', options, data);

            // Connect the JS "select" event to the Python event.
            new QWebChannel(qt.webChannelTransport, function(channel) {
                var widget = channel.objects.widget;
                table.onEvent("select", function (id) {
                    widget._emit_from_js("select", [id]);
                });
            });
        </script>
        ''' % (data_json, columns_json, columns_json)
        self.build()
        block(lambda: self.eval_js('(typeof(table) !== "undefined")'))

    def sort_by(self, name, sort_dir='asc'):
        """Sort by a given variable."""
        logger.log(5, "Sort by `%s` %s.", name, sort_dir)
        self.eval_js('table.sort_("{}", "{}");'.format(name, sort_dir))

    def get_ids(self):
        """Get the list of ids."""
        ids = self.eval_js('table._getIds();')
        return ids

    def get_next_id(self):
        """Get the next non-skipped row id."""
        next_id = self.eval_js('table.getSiblingId(undefined, "next");')
        return int(next_id) if next_id is not None else None

    def get_previous_id(self):
        """Get the previous non-skipped row id."""
        prev_id = self.eval_js('table.getSiblingId(undefined, "previous");')
        return int(prev_id) if prev_id is not None else None

    def next(self):
        """Select the next non-skipped row."""
        self.eval_js('table.moveToSibling(undefined, "next");')

    def previous(self):
        """Select the previous non-skipped row."""
        self.eval_js('table.moveToSibling(undefined, "previous");')

    def select(self, ids):
        """Select some rows in the table."""
        ids = _uniq(ids)
        self.eval_js('table.select_({});'.format(dumps(ids)))

    def add(self, objects):
        self.eval_js('table.add_({});'.format(dumps(objects)))

    def change(self, objects):
        self.eval_js('table.change_({});'.format(dumps(objects)))

    def remove(self, ids):
        self.eval_js('table.remove_({});'.format(dumps(ids)))

    @property
    def selected(self):
        """Currently selected rows."""
        return [int(_) for _ in self.eval_js('table.selected()') or ()]

    @property
    def current_sort(self):
        """Current sort: a tuple `(name, dir)`."""
        sort = self.eval_js('table._currentSort()')
        return None if not sort else tuple(sort)
