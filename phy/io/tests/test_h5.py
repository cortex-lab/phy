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
        h5file.create_dataset('/mygroup/ds1', (10,), dtype=np.int8)
        group.attrs['myattr'] = 123
        return tempfile.filename


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_h5_read():
    with TemporaryDirectory() as tempdir:
        cwd = os.getcwd()
        os.chdir(tempdir)
        filename = _create_test_file()
        with open_h5(filename) as f:
            # TODO
            pass
            # data = f.read('/mydataset')
            # value = f.read_attr('/path/to/node', 'myattr')
        os.chdir(cwd)
