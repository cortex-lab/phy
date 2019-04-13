# -*- coding: utf-8 -*-

"""py.test utilities."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import numpy as np
import os
import warnings

import matplotlib
from pytest import yield_fixture

from phy import add_default_handler
from phy.utils.tempdir import TemporaryDirectory


#------------------------------------------------------------------------------
# Common fixtures
#------------------------------------------------------------------------------

logging.getLogger().setLevel(10)
add_default_handler(5)

# Fix the random seed in the tests.
np.random.seed(2015)

warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


@yield_fixture
def tempdir():
    with TemporaryDirectory() as tempdir:
        yield tempdir


@yield_fixture
def chdir_tempdir():
    curdir = os.getcwd()
    with TemporaryDirectory() as tempdir:
        os.chdir(tempdir)
        yield tempdir
    os.chdir(curdir)
