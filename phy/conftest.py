# -*- coding: utf-8 -*-

"""py.test utilities."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import numpy as np
import warnings

import matplotlib

from phylib import add_default_handler
from phylib.conftest import *  # noqa


#------------------------------------------------------------------------------
# Common fixtures
#------------------------------------------------------------------------------

logger = logging.getLogger('phy')
logger.setLevel(10)
add_default_handler(5, logger=logger)

# Fix the random seed in the tests.
np.random.seed(2019)

warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


def pytest_addoption(parser):
    """Repeat option."""
    parser.addoption('--repeat', action='store', help='Number of times to repeat each test')


def pytest_generate_tests(metafunc):  # pragma: no cover
    # Use --repeat option.
    if metafunc.config.option.repeat is not None:
        count = int(metafunc.config.option.repeat)
        metafunc.fixturenames.append('tmp_ct')
        metafunc.parametrize('tmp_ct', range(count))
