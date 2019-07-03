# -*- coding: utf-8 -*-

"""Test gui."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import os
import shutil

from ..state import GUIState, _gui_state_path, _get_default_state_path
from phylib.utils import Bunch, load_json, save_json

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Test GUI state
#------------------------------------------------------------------------------

class MyClass(object):
    pass


def test_get_default_state_path():
    assert str(_get_default_state_path(MyClass())).endswith(
        os.sep.join(('gui', 'tests', 'static', 'state.json')))


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


def test_gui_state_view_2(tempdir):
    global_path = tempdir / 'global/state.json'
    local_path = tempdir / 'local/state.json'
    data = {'a': {'b': 2, 'c': 3}}

    # Keep the entire dictionary with 'a' key.
    state = GUIState(global_path, local_path=local_path, local_keys=('a.d',))
    state.update(data)
    state.save()

    # Local and global files are identical.
    assert load_json(global_path) == data
    assert load_json(local_path) == {}

    state = GUIState(global_path, local_path=local_path, local_keys=('a.d',))
    assert state == data


def test_gui_state_view_3(tempdir):
    global_path = tempdir / 'global/state.json'
    local_path = tempdir / 'local/state.json'
    data = {'a': {'b': 2, 'c': 3}}

    state = GUIState(global_path, local_path=local_path)
    state.add_local_keys(['a.b'])
    state.update(data)
    state.save()

    assert load_json(global_path) == {'a': {'c': 3}}
    # Only kept key 'b'.
    assert load_json(local_path) == {'a': {'b': 2}}
    # Update the JSON
    save_json(local_path, {'a': {'b': 3}})

    state = GUIState(global_path, local_path=local_path, local_keys=('a.b',))
    data_1 = {'a': {'b': 3, 'c': 3}}
    assert state == data_1
    assert state._local_data == {'a': {'b': 3}}
