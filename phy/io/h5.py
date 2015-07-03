# -*- coding: utf-8 -*-

"""HDF5 input and output."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
import h5py

from ..ext.six import string_types
from ..utils.logging import debug, warn


#------------------------------------------------------------------------------
# HDF5 utility functions
#------------------------------------------------------------------------------

def _split_hdf5_path(path):
    """Return the group and dataset of the path."""
    # Make sure the path starts with a leading slash.
    if not path.startswith('/'):
        raise ValueError(("The HDF5 path '{0:s}' should start with a "
                          "leading slash '/'.").format(path))
    if '//' in path:
        raise ValueError(("There should be no double slash in the HDF5 path "
                          "'{0:s}'.").format(path))
    # Handle the special case '/'.
    if path == '/':
        return '/', ''
    # Temporarily remove the leading '/', we'll add it later (otherwise split
    # and join will mess it up).
    path = path[1:]
    # We split the path by slash and we get the head and tail.
    _split = path.split('/')
    group_path = '/'.join(_split[:-1])
    name = _split[-1]
    # Make some consistency checks.
    assert not group_path.endswith('/')
    assert '/' not in name
    # Finally, we add the leading slash at the beginning of the group path.
    return '/' + group_path, name


def _check_hdf5_path(h5_file, path):
    """Check that an HDF5 path exists in a file."""
    if path not in h5_file:
        raise ValueError("{path} doesn't exist.".format(path=path))


#------------------------------------------------------------------------------
# File class
#------------------------------------------------------------------------------

class File(object):
    def __init__(self, filename, mode=None):
        if mode is None:
            mode = 'r'
        self.filename = filename
        self.mode = mode
        self._h5py_file = None

    # Open and close
    #--------------------------------------------------------------------------

    @property
    def h5py_file(self):
        """Native h5py file handle."""
        return self._h5py_file

    def is_open(self):
        return self._h5py_file is not None

    def open(self, mode=None):
        if mode is not None:
            self.mode = mode
        if not self.is_open():
            self._h5py_file = h5py.File(self.filename, self.mode)

    def close(self):
        if self.is_open():
            self._h5py_file.close()
            self._h5py_file = None

    # Datasets
    #--------------------------------------------------------------------------

    def read(self, path):
        """Read an HDF5 dataset, given its HDF5 path in the file."""
        _check_hdf5_path(self._h5py_file, path)
        return self._h5py_file[path]

    def write(self, path, array=None, dtype=None, shape=None, overwrite=False):
        """Write a NumPy array in the file.

        Parameters
        ----------
        path : str
            Full HDF5 path to the dataset to create.
        array : ndarray
            Array to write in the file.
        dtype : dtype
            If `array` is None, the dtype of the array.
        shape : tuple
            If `array` is None, the shape of the array.
        overwrite : bool
            If False, raise an error if the dataset already exists. Defaults
            to False.

        """
        # Get the group path and the dataset name.
        group_path, dset_name = _split_hdf5_path(path)

        # If the parent group doesn't already exist, create it.
        if group_path not in self._h5py_file:
            self._h5py_file.create_group(group_path)

        group = self._h5py_file[group_path]

        # Check that the dataset does not already exist.
        if path in self._h5py_file:
            if overwrite:
                # Force rewriting the dataset if 'overwrite' is True.
                del self._h5py_file[path]
            else:
                # Otherwise, raise an error.
                raise ValueError(("The dataset '{0:s}' already exists."
                                  ).format(path))

        if array is not None:
            return group.create_dataset(dset_name, data=array)
        else:
            assert dtype
            assert shape
            return group.create_dataset(dset_name, dtype=dtype, shape=shape)

    # Copy and rename
    #--------------------------------------------------------------------------

    def _check_move_copy(self, path, new_path):
        if not self.exists(path):
            raise ValueError("'{0}' doesn't exist.".format(path))
        if self.exists(new_path):
            raise ValueError("'{0}' already exist.".format(new_path))

    def move(self, path, new_path):
        """Move a group or dataset to another location."""
        self._check_move_copy(path, new_path)
        self._h5py_file.move(path, new_path)

    def copy(self, path, new_path):
        """Copy a group or dataset to another location."""
        self._check_move_copy(path, new_path)
        self._h5py_file.copy(path, new_path)

    def delete(self, path):
        """Delete a group or dataset."""
        if not path.startswith('/'):
            path = '/' + path
        if not self.exists(path):
            raise ValueError("'{0}' doesn't exist.".format(path))

        path, name = _split_hdf5_path(path)
        parent = self.read(path)
        del parent[name]

    # Attributes
    #--------------------------------------------------------------------------

    def read_attr(self, path, attr_name):
        """Read an attribute of an HDF5 group."""
        _check_hdf5_path(self._h5py_file, path)
        attrs = self._h5py_file[path].attrs
        if attr_name in attrs:
            try:
                out = attrs[attr_name]
                if isinstance(out, np.ndarray) and out.dtype.kind == 'S':
                    if len(out) == 1:
                        out = out[0].decode('UTF-8')
                return out
            except (TypeError, IOError):
                debug("Unable to read attribute `{}` at `{}`.".format(
                      attr_name, path))
                return
        else:
            raise KeyError("The attribute '{0:s}' ".format(attr_name) +
                           "at `{}` doesn't exist.".format(path))

    def write_attr(self, path, attr_name, value):
        """Write an attribute of an HDF5 group."""
        assert isinstance(path, string_types)
        assert isinstance(attr_name, string_types)
        if value is None:
            value = ''
        # Ensure lists of strings are converted to ASCII arrays.
        if isinstance(value, list):
            if not value:
                value = None
            if value and isinstance(value[0], string_types):
                value = np.array(value, dtype='S')
        # Use string arrays instead of vlen arrays (crash in h5py 2.5.0 win64).
        if isinstance(value, string_types):
            value = np.array([value], dtype='S')
        # Idem: fix crash with boolean attributes on win64.
        if isinstance(value, bool):
            value = int(value)
        # If the parent group doesn't already exist, create it.
        if path not in self._h5py_file:
            self._h5py_file.create_group(path)
        try:
            self._h5py_file[path].attrs[attr_name] = value
            # debug("Write `{}={}` at `{}`.".format(attr_name,
            #                                       str(value), path))
        except TypeError:
            warn("Unable to write attribute `{}={}` at `{}`.".format(
                 attr_name, value, path))

    def attrs(self, path='/'):
        """Return the list of attributes at the given path."""
        if path in self._h5py_file:
            return sorted(self._h5py_file[path].attrs)
        else:
            return []

    def has_attr(self, path, attr_name):
        """Return whether an attribute exists at a given path."""
        if path not in self._h5py_file:
            return False
        else:
            return attr_name in self._h5py_file[path].attrs

    # Children
    #--------------------------------------------------------------------------

    def children(self, path='/'):
        """Return the list of children of a given node."""
        return sorted(self._h5py_file[path].keys())

    def groups(self, path='/'):
        """Return the list of groups under a given node."""
        return [key for key in self.children(path)
                if isinstance(self._h5py_file[path + '/' + key],
                              h5py.Group)]

    def datasets(self, path='/'):
        """Return the list of datasets under a given node."""
        return [key for key in self.children(path)
                if isinstance(self._h5py_file[path + '/' + key],
                              h5py.Dataset)]

    # Miscellaneous properties
    #--------------------------------------------------------------------------

    def exists(self, path):
        return path in self._h5py_file

    def _print_node_info(self, name, node):
        """Print node information."""
        info = ('/' + name).ljust(50)
        if isinstance(node, h5py.Group):
            pass
        elif isinstance(node, h5py.Dataset):
            info += str(node.shape).ljust(20)
            info += str(node.dtype).ljust(8)
        print(info)

    def describe(self):
        """Display the list of all groups and datasets in the file."""
        if not self.is_open():
            raise IOError("Cannot display file information because the file"
                          " '{0:s}' is not open.".format(self.filename))
        self._h5py_file['/'].visititems(self._print_node_info)

    # Context manager
    #--------------------------------------------------------------------------

    def __contains__(self, path):
        return path in self._h5py_file

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, tb):
        self.close()


def open_h5(filename, mode=None):
    """Open an HDF5 file and return a File instance."""
    file = File(filename, mode=mode)
    file.open()
    return file
