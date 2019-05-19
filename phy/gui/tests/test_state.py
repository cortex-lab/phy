# -*- coding: utf-8 -*-

"""Test gui."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import shutil

from ..state import GUIState, _gui_state_path, _get_default_state_path, _filter_nested_dict
from phylib.utils import Bunch, _load_json, _save_json

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Test GUI state
#------------------------------------------------------------------------------

class MyClass(object):
    pass


def test_get_default_state_path():
    assert str(_get_default_state_path(MyClass())).endswith('gui/tests/static/state.json')


def test_gui_state_view_1(tempdir):
    view = Bunch(name='MyView0')
    path = _gui_state_path('GUI', tempdir)
    state = GUIState(path)
    state.update_view_state(view, dict(hello='world'))
    assert not state.get_view_state(Bunch(name='MyView'))
    assert not state.get_view_state(Bunch(name='MyView (1)'))
    assert state.get_view_state(view) == Bunch(hello='world')
    state.save()

    # Copy the state.json to a "default" location.
    default_path = tempdir / 'state.json'
    shutil.copy(state._path, default_path)
    state._path.unlink()

    logger.info("Create new GUI state.")
    # The default state.json should be automatically copied and loaded.
    state = GUIState(path, default_state_path=default_path)
    assert state.MyView0.hello == 'world'


def test_filter_nested_dict():
    def _assert(d0, d1=None, search_terms=None):
        d1 = d1 if d1 is not None else d0
        assert _filter_nested_dict(d0, search_terms=search_terms) == d1
    _assert({})
    _assert({'a': 1})

    _assert({'a': 1}, {}, search_terms=('b',))
    _assert({'a': 1}, search_terms=('a',))

    data = {'a': {'b': 2, 'c': {'d': 4, 'e': 5}}}
    _assert(data)
    _assert(data, {}, search_terms=('f',))
    # Keep the entire dictionary.
    _assert(data, search_terms=('a',))
    _assert(data, {'a': {'b': 2}}, search_terms=('b',))
    _assert(data, {'a': {'c': {'d': 4, 'e': 5}}}, search_terms=('c',))
    _assert(data, {'a': {'c': {'d': 4}}}, search_terms=('d',))
    _assert(data, {'a': {'c': {'d': 4, 'e': 5}}}, search_terms=('d', 'e'))
    _assert(data, {'a': {'b': 2, 'c': {'e': 5}}}, search_terms=('b', 'e'))


def test_gui_state_view_2(tempdir):
    global_path = tempdir / 'global/state.json'
    local_path = tempdir / 'local/state.json'
    data = {'a': {'b': 2, 'c': {'d': 4, 'e': 5}}}

    # Keep the entire dictionary with 'a' key.
    state = GUIState(global_path, local_path=local_path, local_keys=('a',))
    state.update(data)
    state.save()

    # Local and global files are identical.
    assert _load_json(global_path) == _load_json(local_path)

    state = GUIState(global_path, local_path=local_path, local_keys=('a',))
    assert state == data


def test_gui_state_view_3(tempdir):
    global_path = tempdir / 'global/state.json'
    local_path = tempdir / 'local/state.json'
    data = {'a': {'b': 2, 'c': {'d': 4, 'e': 5}}}

    state = GUIState(global_path, local_path=local_path, local_keys=('b',))
    state.update(data)
    state.save()

    # Global file is not affected by local state.
    assert _load_json(global_path) == state
    # Only kept key 'b'.
    assert _load_json(local_path) == {'a': {'b': 2}}
    # Update the JSON
    _save_json(local_path, {'a': {'b': 3}})

    state = GUIState(global_path, local_path=local_path, local_keys=('b',))
    data_1 = data.copy()
    data_1['a']['b'] = 3
    assert state == data_1
    assert state._local_data == {'a': {'b': 3}}
