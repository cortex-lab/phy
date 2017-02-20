# -*- coding: utf-8 -*-

"""HTML widgets for GUIs."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

from collections import OrderedDict
import json
import logging

from six import text_type

from .qt import WebView, QWebChannel, QVariant, pyqtSlot, block
from phy.utils import EventEmitter
from phy.utils._misc import _CustomEncoder

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

                widget.emit = function(name, arg) {
                    widget._emit_from_js(name, JSON.stringify(arg));
                };
                window.emit = widget.emit;
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


def _to_py(obj):  # pragma: no cover
    if isinstance(obj, QVariant):
        return obj.toPyObject()
    elif isinstance(obj, list):
        return [_to_py(_) for _ in obj]
    elif isinstance(obj, tuple):
        return tuple(_to_py(_) for _ in obj)
    else:
        return obj


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

    def __init__(self, title=''):
        super(Table, self).__init__(title=title)
        self._set_builder()
        self._columns = OrderedDict()
        self._default_sort = (None, None)
        self.build()
        # Make sure the table is fully loaded at initialization.
        block(lambda: self.eval_js('(typeof(window.table) !== "undefined")'))
        self.add_column(lambda _: _, name='id')

    def _set_builder(self):
        b = self.builder
        b.add_style_src('table.css')
        b.add_script_src('tablesort.min.js')
        b.add_script_src('tablesort.number.js')
        b.add_script_src('table.js')
        b.set_body('<table id="{}" class="sort"></table>'.format(
                   self._table_id))
        b.add_script('''
            onWidgetReady(function() {
                window.table = new Table(document.getElementById("%s"));
            });
        ''' % self._table_id)

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
        self.eval_js('window.table.setHeaders({});'.format(data))

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
        self.eval_js('window.table.setData({});'.format(data))

        # Sort.
        if sort_col:
            self.sort_by(sort_col, sort_dir)

    def sort_by(self, name, sort_dir='asc'):
        """Sort by a given variable."""
        logger.log(5, "Sort by `%s` %s.", name, sort_dir)
        self.eval_js('window.table.sortBy("{}", "{}");'.format(name, sort_dir))

    def get_next_id(self):
        """Get the next non-skipped row id."""
        next_id = self.eval_js('window.table.get_next_id();')
        return int(next_id) if next_id is not None else None

    def get_previous_id(self):
        """Get the previous non-skipped row id."""
        previous_id = self.eval_js('window.table.get_previous_id();')
        return int(previous_id) if previous_id is not None else None

    def next(self):
        """Select the next non-skipped row."""
        self.eval_js('window.table.next();')

    def previous(self):
        """Select the previous non-skipped row."""
        self.eval_js('window.table.previous();')

    def select(self, ids, do_emit=True, **kwargs):
        """Select some rows in the table.

        By default, the `select` event is raised, unless `do_emit=False`.

        """
        # Select the rows without emiting the event.
        self.eval_js('window.table.select({}, false);'.format(dumps(ids)))
        if do_emit:
            # Emit the event manually if needed.
            self.emit('select', ids, **kwargs)

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
        return [int(_) for _ in self.eval_js('window.table.selected') or ()]

    @property
    def current_sort(self):
        """Current sort: a tuple `(name, dir)`."""
        return tuple(self.eval_js('window.table.currentSort()') or
                     (None, None))
