# -*- coding: utf-8 -*-

"""Execution context that handles parallel processing and cacheing."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from functools import wraps
import inspect
import logging
import os
from pathlib import Path

from six.moves.cPickle import dump, load

from phylib.utils._misc import (
    _save_json, _load_json,
    _load_pickle, _save_pickle,
    _fullname,)
from .config import phy_config_dir, _ensure_dir_exists

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Context
#------------------------------------------------------------------------------

def _cache_methods(obj, memcached, cached):  # pragma: no cover
    for name in memcached:
        f = getattr(obj, name)
        setattr(obj, name, obj.context.memcache(f))

    for name in cached:
        f = getattr(obj, name)
        setattr(obj, name, obj.context.cache(f))


class Context(object):
    """Handle function cacheing and parallel map with ipyparallel."""
    def __init__(self, cache_dir, ipy_view=None, verbose=0):
        self.verbose = verbose
        # Make sure the cache directory exists.
        self.cache_dir = Path(cache_dir).expanduser()
        if not self.cache_dir.exists():
            logger.debug("Create cache directory `%s`.", self.cache_dir)
            os.makedirs(self.cache_dir)

        # Ensure the memcache directory exists.
        path = self.cache_dir / 'memcache'
        if not path.exists():
            path.mkdir()

        self._set_memory(self.cache_dir)
        self.ipy_view = ipy_view if ipy_view else None
        self._memcache = {}

    def _set_memory(self, cache_dir):
        # Try importing joblib.
        try:
            from joblib import Memory
            self._memory = Memory(
                location=self.cache_dir, mmap_mode=None, verbose=self.verbose)
            logger.debug("Initialize joblib cache dir at `%s`.",
                         self.cache_dir)
        except ImportError:  # pragma: no cover
            logger.warning(
                "Joblib is not installed. Install it with `conda install joblib`.")
            self._memory = None

    def cache(self, f):
        """Cache a function using the context's cache directory."""
        if self._memory is None:  # pragma: no cover
            logger.debug("Joblib is not installed: skipping cacheing.")
            return f
        assert f
        # NOTE: discard self in instance methods.
        if 'self' in inspect.getfullargspec(f).args:
            ignore = ['self']
        else:
            ignore = None
        disk_cached = self._memory.cache(f, ignore=ignore)
        return disk_cached

    def load_memcache(self, name):
        # Load the memcache from disk, if it exists.
        path = self.cache_dir / 'memcache' / (name + '.pkl')
        if path.exists():
            logger.debug("Load memcache for `%s`.", name)
            with open(path, 'rb') as fd:
                cache = load(fd)
        else:
            cache = {}
        self._memcache[name] = cache
        return cache

    def save_memcache(self):
        for name, cache in self._memcache.items():
            path = self.cache_dir / 'memcache' / (name + '.pkl')
            logger.debug("Save memcache for `%s`.", name)
            with open(path, 'wb') as fd:
                dump(cache, fd)

    def memcache(self, f):
        """Cache a function in memory using an internal dictionary."""
        name = _fullname(f)
        cache = self.load_memcache(name)

        @wraps(f)
        def memcached(*args):
            """Cache the function in memory."""
            # The arguments need to be hashable. Much faster than using hash().
            h = args
            out = cache.get(h, None)
            if out is None:
                out = f(*args)
                cache[h] = out
            return out
        return memcached

    def _get_path(self, name, location, file_ext='.json'):
        if location == 'local':
            return self.cache_dir / (name + file_ext)
        elif location == 'global':
            return phy_config_dir() / (name + file_ext)

    def save(self, name, data, location='local', kind='json'):
        """Save a dictionary in a JSON file within the cache directory."""
        file_ext = '.json' if kind == 'json' else '.pkl'
        path = self._get_path(name, location, file_ext=file_ext)
        _ensure_dir_exists(path.parent)
        logger.debug("Save data to `%s`.", path)
        if kind == 'json':
            _save_json(path, data)
        else:
            _save_pickle(path, data)

    def load(self, name, location='local'):
        """Load saved data from the cache directory."""
        path = self._get_path(name, location, file_ext='.json')
        if path.exists():
            return _load_json(path)
        path = self._get_path(name, location, file_ext='.pkl')
        if path.exists():
            return _load_pickle(path)
        logger.debug("The file `%s` doesn't exist.", path)
        return {}

    def __getstate__(self):
        """Make sure that this class is picklable."""
        state = self.__dict__.copy()
        state['_memory'] = None
        return state

    def __setstate__(self, state):
        """Make sure that this class is picklable."""
        self.__dict__ = state
        # Recreate the joblib Memory instance.
        self._set_memory(state['cache_dir'])
