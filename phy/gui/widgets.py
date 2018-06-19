# -*- coding: utf-8 -*-

"""HTML widgets for GUIs."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import json
import logging

from six import text_type

from .qt import WebView, QWebChannel, pyqtSlot, _abs_path
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

    input.search {
        width: 100% !important;
    }

    table tr[data-is_masked='true'] {
        color: #888;
    }
"""


_DEFAULT_SCRIPT = """
    window._onWidgetReady_callbacks = [];

    onWidgetReady = function (callback) {
        window._onWidgetReady_callbacks.push(callback);
    };

    document.addEventListener("DOMContentLoaded", function () {
        new QWebChannel(qt.webChannelTransport, function (channel) {
            var widget = channel.objects.widget;
            window.widget = widget;

            // All phy_events emitted from JS are relayed to
            // Python's _emit_from_js().
            document.addEventListener("phy_event", function (e) {
                console.debug("Emit from JS global: " +
                              e.detail.name + " " + e.detail.data);
                widget._emit_from_js(e.detail.name,
                                     JSON.stringify(e.detail.data));
            });

            // Callbacks on the widget.
            for (let callback of window._onWidgetReady_callbacks) {
                callback(widget);
            }

        });
    });
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
        self.set_html(self.builder.html)

    def view_source(self, callback=None):
        return self.eval_js("document.getElementsByTagName('html')[0].innerHTML",
                            callback=callback)

    # Events
    # -------------------------------------------------------------------------

    def emit(self, *args, **kwargs):
        logger.log(5, "Emit from Python")
        return self._event.emit(*args, **kwargs)

    def connect_(self, *args, **kwargs):
        self._event.connect(*args, **kwargs)

    def unconnect_(self, *args, **kwargs):
        self._event.unconnect(*args, **kwargs)

    # Javascript methods
    # -------------------------------------------------------------------------

    def eval_js(self, expr, callback=None):
        """Evaluate a Javascript expression."""
        return self.page().runJavaScript(expr, callback or (lambda _: _))

    @pyqtSlot(str, str)
    def _emit_from_js(self, name, arg_json):
        logger.log(5, "Emit from Python %s %s.", name, arg_json)
        self.emit(text_type(name), json.loads(text_type(arg_json)))


# -----------------------------------------------------------------------------
# HTML table
# -----------------------------------------------------------------------------

def dumps(o):
    return json.dumps(o, cls=_CustomEncoder)


class Table(HTMLWidget):
    """A sortable table with support for selection."""

    def __init__(self, columns=None, value_names=None, data=None, title=''):
        super(Table, self).__init__(title=title)
        self._init_table(columns=columns, value_names=value_names, data=data)

    def _init_table(self, columns=None, value_names=None, data=None):
        columns = columns or ['id']
        value_names = value_names or columns
        data = data or []

        b = self.builder
        b.set_body_src('index.html')

        self.data = data
        self.columns = columns
        self.value_names = value_names

        self.emit('pre_build')

        data_json = dumps(self.data)
        columns_json = dumps(self.columns)
        value_names_json = dumps(self.value_names)

        b.body += '''
        <script>
            var data = %s;

            var options = {
              valueNames: %s,
              columns: %s,
            };

            var table = new Table('table', options, data);

        </script>
        ''' % (data_json, value_names_json, columns_json)
        self.build()

    def sort_by(self, name, sort_dir='asc'):
        """Sort by a given variable."""
        logger.log(5, "Sort by `%s` %s.", name, sort_dir)
        self.eval_js('table.sort_("{}", "{}");'.format(name, sort_dir))

    def filter(self, text=''):
        logger.log(5, "Filter table with `%s`.", text)
        self.eval_js('table.filter_("{}");'.format(text))

    def get_ids(self, callback=None):
        """Get the list of ids."""
        self.eval_js('table._getIds();', callback=callback)

    def get_next_id(self, callback=None):
        """Get the next non-skipped row id."""
        self.eval_js('table.getSiblingId(undefined, "next");', callback=callback)

    def get_previous_id(self, callback=None):
        """Get the previous non-skipped row id."""
        self.eval_js('table.getSiblingId(undefined, "previous");', callback=callback)

    def next(self):
        """Select the next non-skipped row."""
        self.eval_js('table.moveToSibling(undefined, "next");')

    def previous(self):
        """Select the previous non-skipped row."""
        self.eval_js('table.moveToSibling(undefined, "previous");')

    def select(self, ids):
        """Select some rows in the table."""
        ids = _uniq(ids)
        self.eval_js('table.select({});'.format(dumps(ids)))

    def get(self, id, callback=None):
        self.eval_js('table.get("id", {})[0]["_values"]'.format(id), callback=callback)

    def add(self, objects):
        self.eval_js('table.add_({});'.format(dumps(objects)))

    def change(self, objects):
        self.eval_js('table.change_({});'.format(dumps(objects)))

    def remove(self, ids):
        self.eval_js('table.remove_({});'.format(dumps(ids)))

    def get_selected(self, callback=None):
        """Currently selected rows."""
        self.eval_js('table.selected()', callback=callback)

    def get_current_sort(self, callback=None):
        """Current sort: a tuple `(name, dir)`."""
        self.eval_js('table._currentSort()', callback=callback)
