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
        dset = tempfile.create_dataset("mydataset", (100,),
                                       dtype=np.float32)
        assert dset is not None
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
            assert f is not None
        os.chdir(cwd)
