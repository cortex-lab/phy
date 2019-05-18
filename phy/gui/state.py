# -*- coding: utf-8 -*-

"""Qt dock window."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

try:  # pragma: no cover
    from collections.abc import Mapping  # noqa
except ImportError:  # pragma: no cover
    from collections import Mapping  # noqa
import inspect
import json
import logging
from pathlib import Path
import shutil

from phylib.utils import Bunch, _bunchify, _load_json, _save_json
from phy.utils import _ensure_dir_exists, phy_config_dir

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# GUI state
# -----------------------------------------------------------------------------

def _get_default_state_path(gui):
    """Return the path to the default state.json for a given GUI."""
    gui_path = Path(inspect.getfile(gui.__class__))
    path = gui_path.parent / 'static' / 'state.json'
    return path


def _gui_state_path(gui_name, config_dir=None):
    """Return the path to the GUI state, given the GUI name and the config dir."""
    return Path(config_dir or phy_config_dir()) / gui_name / 'state.json'


def _load_state(path):
    try:
        logger.debug("Load %s for GUIState.", path)
        data = _load_json(str(path))
    except json.decoder.JSONDecodeError as e:  # pragma: no cover
        logger.warning("Error decoding JSON: %s", e)
        data = {}
    return _bunchify(data)


def _filter_nested_dict(value, key=None, search_terms=None):
    """Return a copy of a nested dictionary where only keys belonging to search_terms are kept."""
    # key is None for the root only.
    # Expression used to test whether we keep a key or not.
    keep = lambda k: k is None or (
        (not search_terms or k in search_terms) and not k.startswith('_'))
    # Process leaves.
    if not isinstance(value, Mapping):
        return value if keep(key) else None
    else:
        if key and keep(key):
            # Keep the entire dictionary if the key is to be kept.
            return value
        dupe_node = {}
        for key, val in value.items():
            cur_node = _filter_nested_dict(val, key=key, search_terms=search_terms)
            if cur_node is not None and (not isinstance(cur_node, dict) or cur_node):
                dupe_node[key] = cur_node
        return dupe_node


def _recursive_update(d, u):
    """Recursively update a nested dict with another."""
    # https://stackoverflow.com/a/3233356/1595060
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = _recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class GUIState(Bunch):
    """Represent the state of the GUI: positions of the views and
    all parameters associated to the GUI and views.

    This is automatically loaded from the configuration directory.

    """
    def __init__(self, path, local_path=None, default_state_path=None, local_keys=None, **kwargs):
        super(GUIState, self).__init__(**kwargs)
        self._path = Path(path)
        _ensure_dir_exists(str(self._path.parent))

        self._local_path = Path(local_path) if local_path else None
        self._local_keys = local_keys or ()
        if self._local_path:
            _ensure_dir_exists(str(self._local_path.parent))

        if not default_state_path:
            logger.warning("The default state path %s does not exist.", default_state_path)
        self._default_state_path = default_state_path

        self.load()

    def get_view_state(self, view):
        """Return the state of a view."""
        return self.get(view.name, Bunch())

    def update_view_state(self, view, state):
        """Update the state of a view."""
        name = view.name
        if name not in self:
            self[name] = Bunch()
        self[name].update(state)
        logger.debug("Update GUI state for %s", name)

    def _copy_default_state(self):
        if self._default_state_path and self._default_state_path.exists():
            logger.debug(
                "The GUI state file `%s` doesn't exist, creating a default one...", self._path)
            shutil.copy(self._default_state_path, self._path)
            logger.info("Copied %s to %s.", self._default_state_path, self._path)
        else:
            logger.debug(
                "Could not copy non-existing default state file %s.", self._default_state_path)

    def load(self):
        """Load the state from the JSON file in the config dir."""
        # Make the self._path exists, by copying the default state if necessary.
        if not self._path.exists():
            self._copy_default_state()
        if not self._path.exists():
            return
        self.update(_load_state(self._path))
        # After having loaded the global state, load the local state if it exists.
        # If values already exist, they are updated.
        if self._local_path and self._local_path.exists():
            _recursive_update(self, _load_state(self._local_path))

    @property
    def _global_data(self):
        return {k: v for k, v in self.items() if not k.startswith('_')}

    @property
    def _local_data(self):
        # Select only keys included in self._local_keys.
        return _filter_nested_dict(self, search_terms=self._local_keys)

    def _save_global(self):
        """Save the entire GUIState to the global file."""
        path = self._path
        logger.debug("Save global GUI state to `%s`.", path)
        _save_json(str(path), self._global_data)

    def _save_local(self):
        """Only save fields which keys are in self._local_keys.

        Need to recursively go through the nested dictionaries to get all fields.

        """
        path = self._local_path
        if not self._local_path or not self._local_keys:
            return
        assert self._local_path

        logger.debug("Save local GUI state to `%s`.", path)
        _save_json(str(path), self._local_data)

    def save(self):
        """Save the state to the JSON files in the config dir (global) and local dir (if any)."""
        self._save_global()
        self._save_local()

    def __eq__(self, other):
        """Equality with other dictionary: compare with global data."""
        return other == self._global_data
