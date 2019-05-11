# -*- coding: utf-8 -*-

"""py.test utilities."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import numpy as np
import warnings

import matplotlib

from phy import add_default_handler
from phylib.conftest import *  # noqa


#------------------------------------------------------------------------------
# Common fixtures
#------------------------------------------------------------------------------

logger = logging.getLogger('phy')
logger.setLevel(10)
add_default_handler(5, logger=logger)

# Fix the random seed in the tests.
np.random.seed(2015)

warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
