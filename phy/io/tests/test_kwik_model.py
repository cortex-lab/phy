# -*- coding: utf-8 -*-

"""Tests of Kwik file opening routines."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os

import numpy as np
import h5py
from pytest import raises

from ...utils.tempdir import TemporaryDirectory
from ..h5 import open_h5
from ..kwik_model import KwikModel


#------------------------------------------------------------------------------
# Utility test routines
#------------------------------------------------------------------------------

def _create_test_file():
    with open_h5('_test.kwik', 'w') as tempfile:
        # Create a random dataset using h5py directly.
        h5file = tempfile.h5py_file
        h5file.create_group('/recording')
        return tempfile.filename


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_kwik_open():
    with TemporaryDirectory() as tempdir:
        # Save the currrent working directory.
        cwd = os.getcwd()
        # Change to the temporary directory.
        os.chdir(tempdir)
        # Create the test HDF5 file in the temporary directory.
        filename = _create_test_file()

        # Test implicit open() method.
        k = KwikModel(filename, channel_group=1, recording=0)

        assert k.recording == 0
        assert k.channel_group == 1

        os.chdir(cwd)
