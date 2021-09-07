# -*- coding: utf-8 -*-

"""Test widgets."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from functools import partial
from pathlib import Path
from pytest import fixture, mark

from phylib.utils import connect, unconnect
from phylib.utils.testing import captured_logging
import phy
from .test_qt import _block
from ..widgets import HTMLWidget, Table, Barrier, IPythonView, KeyValueWidget


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

def _assert(f, expected):
    _out = []
    f(lambda x: _out.append(x))
    _block(lambda: _out == [expected])


def _wait_until_table_ready(qtbot, table):
    b = Barrier()
    connect(b(1), event='ready', sender=table)

    table.show()
    qtbot.addWidget(table)
    qtbot.waitForWindowShown(table)
    b.wait()


@fixture
def table(qtbot):
    columns = ["id", "count"]
    data = [{"id": i,
             "count": 100 - 10 * i,
             "float": float(i),
             "is_masked": True if i in (2, 3, 5) else False,
             } for i in range(10)]
    table = Table(
        columns=columns,
        value_names=['id', 'count', {'data': ['is_masked']}],
        data=data)
    _wait_until_table_ready(qtbot, table)

    yield table

    table.close()


#------------------------------------------------------------------------------
# Test widgets
#------------------------------------------------------------------------------

def test_widget_empty(qtbot):
    widget = HTMLWidget()
    widget.build()
    widget.show()
    qtbot.addWidget(widget)
    qtbot.waitForWindowShown(widget)
    widget.close()


def test_widget_html(qtbot):
    widget = HTMLWidget()
    widget.builder.add_style('html, body, p {background-color: purple;}')
    path = Path(__file__).parent.parent / 'static/styles.css'
    widget.builder.add_style_src(path)
    widget.builder.add_header('<!-- comment -->')
    widget.builder.set_body('Hello world!')
    widget.build()
    widget.show()
    qtbot.addWidget(widget)
    qtbot.waitForWindowShown(widget)
    _block(lambda: 'Hello world!' in str(widget.html))

    _out = []

    widget.view_source(lambda x: _out.append(x))
    _block(lambda: _out[0].startswith('<head>') if _out else None)

    # qtbot.stop()
    widget.close()


def test_widget_javascript_1(qtbot):
    widget = HTMLWidget()
    widget.builder.add_script('var number = 1;')
    widget.build()
    widget.show()
    qtbot.addWidget(widget)
    qtbot.waitForWindowShown(widget)
    _block(lambda: widget.html is not None)

    _out = []

    def _callback(res):
        _out.append(res)

    widget.eval_js('number', _callback)
    _block(lambda: _out == [1])

    # Test logging from JS.
    with captured_logging('phy.gui') as buf:
        widget.eval_js('console.warn("hello world!");')
        _block(lambda: 'hello world!' in buf.getvalue().lower())

    # qtbot.stop()
    widget.close()


@mark.parametrize("event_name", ('select', 'nodebounce'))
def test_widget_javascript_debounce(qtbot, event_name):
    phy.gui.qt.Debouncer.delay = 300

    widget = HTMLWidget(debounce_events=('select',))
    widget.build()
    widget.show()
    qtbot.addWidget(widget)
    qtbot.waitForWindowShown(widget)
    _block(lambda: widget.html is not None)

    event_code = lambda i: r'''
    var event = new CustomEvent("phy_event", {detail: {name: '%s', data: {'i': %s}}});
    document.dispatchEvent(event);
    ''' % (event_name, i)

    _l = []

    def f(sender, *args):
        _l.append(args)
    connect(f, sender=widget, event=event_name)

    for i in range(5):
        widget.eval_js(event_code(i))
        qtbot.wait(10)
    qtbot.wait(500)

    assert len(_l) == (2 if event_name == 'select' else 5)

    # qtbot.stop()
    widget.close()

    phy.gui.qt.Debouncer.delay = 1


#------------------------------------------------------------------------------
# Test key value widget
#------------------------------------------------------------------------------

def test_key_value_1(qtbot):
    widget = KeyValueWidget()
    widget.show()

    qtbot.addWidget(widget)
    qtbot.waitForWindowShown(widget)

    widget.add_pair("my text", "some text")
    widget.add_pair("my text multiline", "some\ntext", 'multiline')
    widget.add_pair("my float", 3.5)
    widget.add_pair("my int", 3)
    widget.add_pair("my bool", True)
    widget.add_pair("my list", [1, 5])

    widget.get_widget('my bool').setChecked(False)
    widget.get_widget('my list[0]').setValue(2)

    assert widget.to_dict() == {
        'my text': 'some text', 'my text multiline': 'some\ntext',
        'my float': 3.5, 'my int': 3, 'my bool': False, 'my list': [2, 5]}

    # qtbot.stop()
    widget.close()


#------------------------------------------------------------------------------
# Test IPython view
#------------------------------------------------------------------------------

@mark.filterwarnings("ignore")
def test_ipython_view_1(qtbot):
    view = IPythonView()
    view.show()
    view.start_kernel()
    view.stop()
    qtbot.wait(10)
    view.close()


@mark.filterwarnings("ignore")
def test_ipython_view_2(qtbot, tempdir):
    from ..gui import GUI
    gui = GUI(config_dir=tempdir)
    gui.set_default_actions()

    view = IPythonView()
    view.show()

    view.attach(gui)  # start the kernel and inject the GUI

    gui.show()
    view.dock.close()
    qtbot.wait(10)
    gui.close()
    qtbot.wait(10)


#------------------------------------------------------------------------------
# Test table
#------------------------------------------------------------------------------

def test_barrier_1(qtbot, table):
    table.select([1])

    b = Barrier()
    table.get_selected(b(1))
    table.get_next_id(b(2))
    assert not b.have_all_finished()

    @b.after_all_finished
    def after():
        assert b.result(1)[0][0] == [1]
        assert b.result(2)[0][0] == 4

    b.wait()
    assert b.result(1) and b.result(2)


def test_table_empty_1(qtbot):
    table = Table()
    _wait_until_table_ready(qtbot, table)
    assert table.debouncer
    table.close()


def test_table_invalid_column(qtbot):
    table = Table(data=[{'id': 0, 'a': 'b'}], columns=['id', 'u'])
    table.show()
    qtbot.addWidget(table)
    qtbot.waitForWindowShown(table)
    table.close()


def test_table_0(qtbot, table):
    _assert(table.get_selected, [])


def test_table_1(qtbot, table):

    assert table.is_ready()

    table.select([1, 2])
    _assert(table.get_selected, [1, 2])


def test_table_scroll(qtbot, table):
    table.add([{'id': 1000 + i, 'count': i} for i in range(1000)])
    qtbot.wait(50)
    table.scroll_to(1400)


def test_table_busy(qtbot, table):
    table.select([1, 2])
    table.set_busy(True)
    _l = []

    def callback(out):
        _l.append(out)

    table.eval_js('table.debouncer.isBusy', callback=callback)
    _block(lambda: _l == [True])
    table.set_busy(False)


def test_table_duplicates(qtbot, table):
    table.select([1, 1])
    _assert(table.get_selected, [1])


def test_table_nav_first_1(qtbot, table):
    table.next()
    _assert(table.get_selected, [0])
    _assert(table.get_next_id, 1)


def test_table_nav_first_2(qtbot, table):
    table.first()
    _assert(table.get_selected, [0])
    _assert(table.get_next_id, 1)


def test_table_nav_last(qtbot, table):
    table.previous()
    _assert(table.get_selected, [0])
    _assert(table.get_previous_id, None)

    table.first()
    qtbot.wait(100)

    table.last()
    qtbot.wait(100)


def test_table_nav_0(qtbot, table):
    table.select([4])

    table.next()
    _assert(table.get_selected, [6])

    table.previous()
    _assert(table.get_selected, [4])


def test_table_nav_1(qtbot, table):
    _sel = []

    @connect(sender=table)
    def on_some_event(sender, items, **kwargs):
        _sel.append(items)

    table.eval_js('table.emit("some_event", 123);')

    _block(lambda: _sel == [123])

    unconnect(on_some_event)


def test_table_sort(qtbot, table):
    table.select([1])
    table.next()
    table.next()
    _assert(table.get_selected, [6])

    _l = []

    @connect(sender=table)
    def on_table_sort(sender, row_ids):
        _l.append(row_ids)

    # Sort by count decreasing, and check that 0 (count 100) comes before
    # 1 (count 90). This checks that sorting works with number).
    table.sort_by('count', 'asc')

    _assert(table.get_current_sort, ['count', 'asc'])
    _assert(table.get_selected, [6])
    _assert(table.get_ids, list(range(9, -1, -1)))

    table.next()
    _assert(table.get_selected, [4])

    table.sort_by('count', 'desc')
    _assert(table.get_ids, list(range(10)))

    assert _l == [list(range(9, -1, -1)), list(range(10))]


def test_table_remove_all(qtbot, table):
    table.remove_all()
    _assert(table.get_ids, [])


def test_table_remove_all_and_add_1(qtbot, table):
    table.remove_all_and_add([])
    _assert(table.get_ids, [])


def test_table_remove_all_and_add_2(qtbot, table):
    table.remove_all_and_add({"id": 1000})
    _assert(table.get_ids, [1000])


def test_table_add_change_remove(qtbot, table):
    _assert(table.get_ids, list(range(10)))

    table.add({'id': 100, 'count': 1000})
    _assert(table.get_ids, list(range(10)) + [100])

    table.remove([0, 1])
    _assert(table.get_ids, list(range(2, 10)) + [100])

    _assert(partial(table.get, 100), {'id': 100, 'count': 1000})
    table.change([{'id': 100, 'count': 2000}])
    _assert(partial(table.get, 100), {'id': 100, 'count': 2000})


def test_table_change_and_sort_1(qtbot, table):
    table.change([{'id': 5, 'count': 1000}])
    _assert(table.get_ids, list(range(10)))


def test_table_change_and_sort_2(qtbot, table):
    table.sort_by('count', 'asc')
    _assert(table.get_ids, list(range(9, -1, -1)))

    # Check that the table is automatically resorted after a change.
    table.change([{'id': 5, 'count': 1000}])
    _assert(table.get_ids, [9, 8, 7, 6, 4, 3, 2, 1, 0, 5])


def test_table_filter(qtbot, table):
    table.filter("id == 5")
    _assert(table.get_ids, [5])

    table.filter("count == 80")
    _assert(table.get_ids, [2])

    table.filter()
    _assert(table.get_ids, list(range(10)))
