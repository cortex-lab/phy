# -*- coding: utf-8 -*-

"""Tests of HDF5 routines."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os

import numpy as np
try:
    import h5py
except ImportError as exception:
    # TODO: logging.
    raise exception

from ...utils.tempdir import TemporaryDirectory
from ..h5 import open_h5


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

        os.chdir(cwd)
