# -*- coding: utf-8 -*-

"""Tests of HDF5 routines."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os

import numpy as np
import h5py
from nose.tools import assert_raises

from ...utils.tempdir import TemporaryDirectory
from ..h5 import open_h5, _split_hdf5_path


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

def setup():
    pass


def teardown():
    pass


#------------------------------------------------------------------------------
# Utility test routines
#------------------------------------------------------------------------------

def _create_test_file():
    with open_h5('_test.h5', 'w') as tempfile:
        # Create a random dataset using h5py directly.
        h5file = tempfile.h5py_file
        h5file.create_dataset('ds1', (10,), dtype=np.float32)
        group = h5file.create_group('/mygroup')
        h5file.create_dataset('/mygroup/ds2', (10,), dtype=np.int8)
        group.attrs['myattr'] = 123
        return tempfile.filename


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_split_hdf5_path():
    # The path should always start with a leading '/'.
    assert_raises(ValueError, _split_hdf5_path, '')
    assert_raises(ValueError, _split_hdf5_path, 'path')

    h, t = _split_hdf5_path('/')
    assert (h == '/') and (t == '')

    h, t = _split_hdf5_path('/path')
    assert (h == '/') and (t == 'path')

    h, t = _split_hdf5_path('/path/')
    assert (h == '/path') and (t == '')

    h, t = _split_hdf5_path('/path/to')
    assert (h == '/path') and (t == 'to')

    h, t = _split_hdf5_path('/path/to/')
    assert (h == '/path/to') and (t == '')

    # Check that invalid paths raise errors.
    assert_raises(ValueError, _split_hdf5_path, 'path/')
    assert_raises(ValueError, _split_hdf5_path, '/path//')
    assert_raises(ValueError, _split_hdf5_path, '/path//to')


def test_h5_read():
    with TemporaryDirectory() as tempdir:
        # Save the currrent working directory.
        cwd = os.getcwd()
        # Change to the temporary directory.
        os.chdir(tempdir)
        # Create the test HDF5 file in the temporary directory.
        filename = _create_test_file()

        # Open the test HDF5 file.
        with open_h5(filename) as f:

            # Check dataset ds1.
            ds1 = f.read('/ds1')[:]
            assert isinstance(ds1, np.ndarray)
            assert ds1.shape == (10,)
            assert ds1.dtype == np.float32

            # Check dataset ds2.
            ds2 = f.read('/mygroup/ds2')[:]
            assert isinstance(ds2, np.ndarray)
            assert ds2.shape == (10,)
            assert ds2.dtype == np.int8

            # Check HDF5 group attribute.
            value = f.read_attr('/mygroup', 'myattr')
            assert value == 123

            # Check that errors are raised when the paths are invalid.
            assert_raises(Exception, f.read, '//path')
            assert_raises(Exception, f.read, '/path//')
            assert_raises(KeyError, f.read, '/nonexistinggroup')
            assert_raises(KeyError, f.read, '/nonexistinggroup/ds34')

        os.chdir(cwd)


def test_h5_write():
    with TemporaryDirectory() as tempdir:
        # Save the currrent working directory.
        cwd = os.getcwd()
        # Change to the temporary directory.
        os.chdir(tempdir)
        # Create the test HDF5 file in the temporary directory.
        filename = _create_test_file()

        # Create some array.
        temp_array = np.zeros(10, dtype=np.float32)

        # Open the test HDF5 file in read-only mode (the default) and
        # try to write in it. This should raise an exception.
        with open_h5(filename) as f:
            assert_raises(Exception, lambda: f.write('/ds1', temp_array))

        # Open the test HDF5 file in read/write mode and
        # try to write in an existing dataset.
        with open_h5(filename, 'a') as f:
            # This raises an exception because the file already exists,
            # and by default this is forbidden.
            assert_raises(ValueError, lambda: f.write('/ds1', temp_array))
            # This works, though, because we force overwriting the dataset.
            f.write('/ds1', temp_array, overwrite=True)

        os.chdir(cwd)
