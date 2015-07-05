# -*- coding: utf-8 -*-

"""Settings."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os
import os.path as op

from .logging import debug, warn
from ._misc import _load_json, _save_json, _read_python


#------------------------------------------------------------------------------
# Settings
#------------------------------------------------------------------------------

class BaseSettings(object):
    """Store key-value pairs."""
    def __init__(self):
        self._store = {}
        self._to_save = {}

    def __getitem__(self, key):
        return self._store[key]

    def __setitem__(self, key, value):
        self._store[key] = value
        self._to_save[key] = value

    def __contains__(self, key):
        return key in self._store

    def __repr__(self):
        return self._store.__repr__()

    def keys(self):
        """List of settings keys."""
        return self._store.keys()

    def _update(self, d):
        for k, v in d.items():
            if isinstance(v, dict) and k in self._store:
                # Update instead of overwrite settings dictionaries.
                self._store[k].update(v)
            else:
                self._store[k] = v

    def _try_load_json(self, path):
        try:
            self._update(_load_json(path))
            debug("Loaded internal settings file "
                  "from `{}`.".format(path))
            return True
        except Exception as e:
            warn("Unable to read the internal settings. "
                 "You may want to delete '{0}'.\n{1}".format(path, str(e)))

    def _try_load_python(self, path):
        try:
            self._update(_read_python(path))
            debug("Loaded internal settings file "
                  "from `{}`.".format(path))
            return True
        except Exception as e:
            warn("Unable to read the settings file "
                 "'{0}':\n{1}".format(path, str(e)))

    def load(self, path):
        """Load a settings file."""
        path = op.realpath(path)
        has_ext = op.splitext(path)[1] != ''
        if not op.exists(path):
            debug("Creating empty settings file: {}.".format(path))
            # Extension => Python, so we can write a comment.
            if has_ext:
                with open(path, 'a') as f:
                    f.write("# Settings file. Refer to phy's documentation "
                            "for more details.\n")
            return
        # Try JSON first, then Python.
        if not has_ext:
            if self._try_load_json(path):
                return
        self._try_load_python(path)

    def save(self, path):
        """Save the settings to a JSON file."""
        path = op.realpath(path)
        try:
            _save_json(path, self._to_save)
            debug("Saved internal settings file "
                  "to `{}`.".format(path))
        except Exception as e:
            warn("Unable to save the internal settings file "
                 "from `{}`:\n{}".format(path, str(e)))
        self._to_save = {}


class Settings(object):
    """Manage default, user-wide, and experiment-wide settings."""

    def __init__(self, phy_user_dir=None, default_path=None):
        self.phy_user_dir = phy_user_dir
        if self.phy_user_dir:
            _ensure_dir_exists(self.phy_user_dir)

        self._default_path = default_path

        self._bs = BaseSettings()
        self._load_user_settings()

    def _load_user_settings(self):
        # Load phy's defaults.
        if self._default_path:
            self._bs.load(self._default_path)

        if not self.phy_user_dir:
            return

        # Load the user defaults.
        self.user_settings_path = op.join(self.phy_user_dir,
                                          'user_settings.py')
        self._bs.load(self.user_settings_path)

        # Load the user's internal settings.
        self.internal_settings_path = op.join(self.phy_user_dir,
                                              'internal_settings')
        self._bs.load(self.internal_settings_path)

    def on_open(self, path):
        """Initialize settings when loading an experiment."""
        if path is None:
            debug("Unable to initialize the settings for unspecified "
                  "model path.")
            return
        # Get the experiment settings path.
        path = op.realpath(op.expanduser(path))
        self.exp_path = path
        self.exp_name = op.splitext(op.basename(path))[0]
        self.exp_dir = op.dirname(path)
        self.exp_settings_dir = op.join(self.exp_dir, self.exp_name + '.phy')
        self.exp_settings_path = op.join(self.exp_settings_dir,
                                         'user_settings.py')
        _ensure_dir_exists(self.exp_settings_dir)

        # Load experiment-wide settings.
        self._load_user_settings()
        self._bs.load(self.exp_settings_path)

    def save(self):
        """Save settings to an internal settings file."""
        self._bs.save(self.internal_settings_path)

    def get(self, key, default=None):
        """Return a settings value."""
        if key in self:
            return self[key]
        else:
            return default

    def __getitem__(self, key):
        return self._bs[key]

    def __setitem__(self, key, value):
        self._bs[key] = value

    def __contains__(self, key):
        return key in self._bs

    def __repr__(self):
        return "<Settings {}>".format(self._bs.__repr__())

    def keys(self):
        """Return the list of settings keys."""
        return self._bs.keys()


#------------------------------------------------------------------------------
# Config
#------------------------------------------------------------------------------

_PHY_USER_DIR_NAME = '.phy'


def _phy_user_dir():
    """Return the absolute path to the phy user directory."""
    home = op.expanduser("~")
    path = op.realpath(op.join(home, _PHY_USER_DIR_NAME))
    return path


def _ensure_dir_exists(path):
    if not op.exists(path):
        os.makedirs(path)
