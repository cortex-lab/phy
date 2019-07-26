# -*- coding: utf-8 -*-

"""Qt dock window."""


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

try:  # pragma: no cover
    from collections.abc import Mapping  # noqa
except ImportError:  # pragma: no cover
    from collections import Mapping  # noqa
from copy import deepcopy
import inspect
import json
import logging
from pathlib import Path
import shutil

from phylib.utils import Bunch, _bunchify, load_json, save_json
from phy.utils import ensure_dir_exists, phy_config_dir

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
    """Load a GUI state from a JSON file."""
    try:
        logger.debug("Load %s for GUIState.", path)
        data = load_json(str(path))
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


def _get_local_data(d, local_keys):
    """Return the Bunch of local data from a full GUI state, and a list of local keys
    of the form `ViewName.field_name`."""
    out = Bunch()
    for key in local_keys:
        key1, key2 = key.split('.')
        val = d.get(key1, {}).get(key2, None)
        # Discard None values
        if val is None:
            continue
        if key1 not in out:
            out[key1] = Bunch()
        out[key1][key2] = val
    return out


def _get_global_data(d, local_keys):
    """Remove the local keys from the GUI state."""
    # d = deepcopy(_filter_nested_dict(d))  # remove private fields
    d = deepcopy(_filter_nested_dict(d))
    for key in local_keys:
        key1, key2 = key.split('.')
        # Remove that key.
        if key1 in d:
            d[key1].pop(key2, None)
    return d


class GUIState(Bunch):
    """Represent the state of the GUI: positions of the views and all parameters associated
    to the GUI and views. Derive from `Bunch`, which itself derives from `dict`.

    The GUI state is automatically loaded from the user configuration directory.
    The default path is `~/.phy/GUIName/state.json`.

    The global GUI state is common to all instances of the GUI.
    The local GUI state is specific to an instance of the GUI, for example a given dataset.

    Constructor
    -----------

    path : str or Path
        The path to the JSON file containing the global GUI state.
    local_path : str or Path
        The path to the JSON file containing the local GUI state.
    default_state_path : str or Path
        The path to the default JSON file provided in the library.
    local_keys : list
        A list of strings `key1.key2` of the elements of the GUI state that should only be saved
        in the local state, and not the global state.

    """
    def __init__(
            self, path=None, local_path=None, default_state_path=None, local_keys=None, **kwargs):
        super(GUIState, self).__init__(**kwargs)
        self._path = Path(path) if path else None
        if self._path:
            ensure_dir_exists(str(self._path.parent))
        self._local_path = Path(local_path) if local_path else None
        self._local_keys = local_keys or []  # A list of strings with full paths key1.key2
        if self._local_path:
            ensure_dir_exists(str(self._local_path.parent))

        if default_state_path:
            default_state_path = Path(default_state_path)
        self._default_state_path = default_state_path

        self.load()

    def get_view_state(self, view):
        """Return the state of a view instance."""
        return self.get(view.name, Bunch())

    def update_view_state(self, view, state):
        """Update the state of a view instance.

        Parameters
        ----------

        view : View instance
        state : Bunch instance

        """
        name = view.name
        if name not in self:
            self[name] = Bunch()
        self[name].update(state)
        logger.debug("Update GUI state for %s", name)

    def _copy_default_state(self):
        """Copy the default GUI state to the user directory."""

        if self._default_state_path and self._default_state_path.exists():
            logger.debug(
                "The GUI state file `%s` doesn't exist, creating a default one...", self._path)
            shutil.copy(self._default_state_path, self._path)
            logger.info("Copied %s to %s.", self._default_state_path, self._path)
        elif self._default_state_path:  # pragma: no cover
            logger.warning(
                "Could not copy non-existing default state file %s.", self._default_state_path)

    def add_local_keys(self, keys):
        """Add local keys."""
        self._local_keys.extend([k for k in keys if k not in self._local_keys])

    def load(self):
        """Load the state from the JSON file in the config dir."""
        # Make the self._path exists, by copying the default state if necessary.
        if not self._path:
            return
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
        """Select non-private fields in the GUI state."""
        return _get_global_data(self, self._local_keys)

    @property
    def _local_data(self):
        """Select fields for the local GUI state."""
        # Select only keys included in self._local_keys.
        return _get_local_data(self, self._local_keys)

    def _save_global(self):
        """Save the entire GUIState to the global file."""
        path = self._path
        logger.debug("Save global GUI state to `%s`.", path)
        save_json(str(path), self._global_data)

    def _save_local(self):
        """Only save local fields (keys are in self._local_keys).

        Need to recursively go through the nested dictionaries to get all fields.

        """
        path = self._local_path
        if not self._local_path or not self._local_keys:
            return
        assert self._local_path

        logger.debug("Save local GUI state to `%s`.", path)
        save_json(str(path), self._local_data)

    def save(self):
        """Save the state to the JSON files in the config dir (global) and local dir (if any)."""
        # Save the GUI state version.
        self._save_global()
        self._save_local()

    def __eq__(self, other):
        """Equality with other dictionary: compare with global data."""
        return _filter_nested_dict(other) == _filter_nested_dict(self)
