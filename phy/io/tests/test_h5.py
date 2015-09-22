# -*- coding: utf-8 -*-

"""Tests of HDF5 routines."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op

import numpy as np
from numpy.testing import assert_array_equal as ae
from pytest import raises

from ...utils.testing import captured_output
from ..h5 import open_h5, _split_hdf5_path


#------------------------------------------------------------------------------
# Utility test routines
#------------------------------------------------------------------------------

def _create_test_file(dirpath):
    filename = op.join(dirpath, '_test.h5')
    with open_h5(filename, 'w') as tempfile:
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
    with raises(ValueError):
        _split_hdf5_path('')
    with raises(ValueError):
        _split_hdf5_path('path')

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
    with raises(ValueError):
        _split_hdf5_path('path/')
    with raises(ValueError):
        _split_hdf5_path('/path//')
    with raises(ValueError):
        _split_hdf5_path('/path//to')


def test_h5_read(tempdir):
    # Create the test HDF5 file in the temporary directory.
    filename = _create_test_file(tempdir)

    # Test close() method.
    f = open_h5(filename)
    assert f.is_open()
    f.close()
    assert not f.is_open()
    with raises(IOError):
        f.describe()

    # Open the test HDF5 file.
    with open_h5(filename) as f:
        assert f.is_open()

        assert f.children() == ['ds1', 'mygroup']
        assert f.groups() == ['mygroup']
        assert f.datasets() == ['ds1']
        assert f.attrs('/mygroup') == ['myattr']
        assert f.attrs('/mygroup_nonexisting') == []

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
        assert f.has_attr('/mygroup', 'myattr')
        assert not f.has_attr('/mygroup', 'myattr_bis')
        assert not f.has_attr('/mygroup_bis', 'myattr_bis')

        # Check that errors are raised when the paths are invalid.
        with raises(Exception):
            f.read('//path')
        with raises(Exception):
            f.read('/path//')
        with raises(ValueError):
            f.read('/nonexistinggroup')
        with raises(ValueError):
            f.read('/nonexistinggroup/ds34')

    assert not f.is_open()


def test_h5_append(tempdir):
    # Create the test HDF5 file in the temporary directory.
    filename = _create_test_file(tempdir)

    with open_h5(filename, 'a') as f:
        f.write('/ds_empty', dtype=np.float32, shape=(10, 2))
        arr = f.read('/ds_empty')
        arr[:5, 0] = 1

    with open_h5(filename, 'r') as f:
        arr = f.read('/ds_empty')[...]
        assert np.all(arr[:5, 0] == 1)


def test_h5_write(tempdir):
    # Create the test HDF5 file in the temporary directory.
    filename = _create_test_file(tempdir)

    # Create some array.
    temp_array = np.zeros(10, dtype=np.float32)

    # Open the test HDF5 file in read-only mode (the default) and
    # try to write in it. This should raise an exception.
    with open_h5(filename) as f:
        with raises(Exception):
            f.write('/ds1', temp_array)

    # Open the test HDF5 file in read/write mode and
    # try to write in an existing dataset.
    with open_h5(filename, 'a') as f:
        # This raises an exception because the file already exists,
        # and by default this is forbidden.
        with raises(ValueError):
            f.write('/ds1', temp_array)

        # This works, though, because we force overwriting the dataset.
        f.write('/ds1', temp_array, overwrite=True)
        ae(f.read('/ds1'), temp_array)

        # Write a new array.
        f.write('/ds2', temp_array)
        ae(f.read('/ds2'), temp_array)

        # Write a new array in a nonexistent group.
        f.write('/ds3/ds4/ds5', temp_array)
        ae(f.read('/ds3/ds4/ds5'), temp_array)


def test_h5_describe(tempdir):
    # Create the test HDF5 file in the temporary directory.
    filename = _create_test_file(tempdir)

    # Open the test HDF5 file.
    with open_h5(filename) as f:
        with captured_output() as (out, err):
            f.describe()
    output = out.getvalue().strip()
    output_lines = output.split('\n')
    assert len(output_lines) == 3


def test_h5_move(tempdir):
    # Create the test HDF5 file in the temporary directory.
    filename = _create_test_file(tempdir)

    with open_h5(filename, 'a') as f:

        # Test dataset move.
        assert f.exists('ds1')
        arr = f.read('ds1')[:]
        assert len(arr) == 10
        f.move('ds1', 'ds1_new')
        assert not f.exists('ds1')
        assert f.exists('ds1_new')
        arr_new = f.read('ds1_new')[:]
        assert len(arr_new) == 10
        ae(arr, arr_new)

        # Test group move.
        assert f.exists('mygroup/ds2')
        arr = f.read('mygroup/ds2')
        f.move('mygroup', 'g/mynewgroup')
        assert not f.exists('mygroup')
        assert f.exists('g/mynewgroup')
        assert f.exists('g/mynewgroup/ds2')
        arr_new = f.read('g/mynewgroup/ds2')
        ae(arr, arr_new)


def test_h5_copy(tempdir):
    # Create the test HDF5 file in the temporary directory.
    filename = _create_test_file(tempdir)

    with open_h5(filename, 'a') as f:

        # Test dataset copy.
        assert f.exists('ds1')
        arr = f.read('ds1')[:]
        assert len(arr) == 10
        f.copy('ds1', 'ds1_new')
        assert f.exists('ds1')
        assert f.exists('ds1_new')
        arr_new = f.read('ds1_new')[:]
        assert len(arr_new) == 10
        ae(arr, arr_new)

        # Test group copy.
        assert f.exists('mygroup/ds2')
        arr = f.read('mygroup/ds2')
        f.copy('mygroup', 'g/mynewgroup')
        assert f.exists('mygroup')
        assert f.exists('g/mynewgroup')
        assert f.exists('g/mynewgroup/ds2')
        arr_new = f.read('g/mynewgroup/ds2')
        ae(arr, arr_new)


def test_h5_delete(tempdir):
    # Create the test HDF5 file in the temporary directory.
    filename = _create_test_file(tempdir)

    with open_h5(filename, 'a') as f:

        # Test dataset delete.
        assert f.exists('ds1')
        with raises(ValueError):
            f.delete('a')
        f.delete('ds1')
        assert not f.exists('ds1')

        # Test group delete.
        assert f.exists('mygroup/ds2')
        f.delete('mygroup')
        assert not f.exists('mygroup')
        assert not f.exists('mygroup/ds2')
