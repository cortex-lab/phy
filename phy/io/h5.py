# -*- coding: utf-8 -*-

"""HDF5 input and output."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
try:
    import h5py
except ImportError as exception:
    # TODO: logging.
    raise exception

from ..ext import six


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
    # # Remove eventual trailing slash.
    # if path.endswith('/'):
    #     path = path[:-1]
    # # Now, there should be no more trailing slash.
    # assert path.endswith('/') is False
    # We split the path by slash and we get the head and tail.
    _split = path.split('/')
    group_path = '/'.join(_split[:-1])
    name = _split[-1]
    # Make some consistency checks.
    assert not group_path.endswith('/')
    assert '/' not in name
    # Finally, we add the leading slash at the beginning of the group path.
    return '/' + group_path, name


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

    #--------------------------------------------------------------------------
    # Reading functions
    #--------------------------------------------------------------------------

    def read(self, path):
        """Read an HDF5 dataset, given its HDF5 path in the file."""
        return self._h5py_file[path]

    def read_attr(self, path, attr_name):
        """Read an attribute of an HDF5 group."""
        return self._h5py_file[path].attrs[attr_name]

    #--------------------------------------------------------------------------
    # Writing functions
    #--------------------------------------------------------------------------

    def write(self, path, array, overwrite=False):
        """Write a NumPy array in the file.

        Parameters
        ----------
        path : str
            Full HDF5 path to the dataset to create.
        array : ndarray
            Array to write in the file.
        overwrite : bool
            If False, raise an error if the dataset already exists. Defaults
            to False.

        """
        # Get the group path and the dataset name.
        group_path, dset_name = _split_hdf5_path(path)
        group = self._h5py_file[group_path]
        # Check that the dataset does not already exists.
        if path in self._h5py_file:
            if overwrite:
                # Force rewriting the dataset if 'overwrite' is True.
                del self._h5py_file[path]
            else:
                # Otherwise, raise an error.
                raise ValueError(("The dataset '{0:s}' already exists."
                                  ).format(path))
        group.create_dataset(dset_name, data=array)

    def write_attr(self, path, attr_name, value):
        """Read an attribute of an HDF5 group."""
        self._h5py_file[path].attrs[attr_name] = value

    #--------------------------------------------------------------------------
    # Open and close
    #--------------------------------------------------------------------------

    def is_open(self):
        return self._h5py_file is not None

    def open(self):
        if not self.is_open():
            self._h5py_file = h5py.File(self.filename, self.mode)

    def close(self):
        if self.is_open():
            self._h5py_file.close()
            self._h5py_file = None

    #--------------------------------------------------------------------------
    # Context manager
    #--------------------------------------------------------------------------

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, tb):
        self.close()

    #--------------------------------------------------------------------------
    # Miscellaneous properties
    #--------------------------------------------------------------------------

    @property
    def h5py_file(self):
        """Native h5py file handle."""
        return self._h5py_file

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


def open_h5(filename, mode=None):
    file = File(filename, mode=mode)
    file.open()
    return file
