"""py.test utilities."""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

import os
import logging
import warnings

import matplotlib
import numpy as np
from phylib import add_default_handler
from phylib.conftest import *  # noqa

# ------------------------------------------------------------------------------
# Common fixtures
# ------------------------------------------------------------------------------

logger = logging.getLogger('phy')
logger.setLevel(10)
add_default_handler(5, logger=logger)

os.environ.setdefault('JUPYTER_PLATFORM_DIRS', '1')
warnings.filterwarnings(
    'ignore',
    message='Jupyter is migrating its paths to use standard platformdirs',
    category=DeprecationWarning,
)
warnings.filterwarnings(
    'ignore',
    message=r'tostring\(\) is deprecated\. Use tobytes\(\) instead\.',
    category=DeprecationWarning,
)

# Fix the random seed in the tests.
np.random.seed(2019)

# warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)
# warnings.filterwarnings("ignore", message="numpy.dtype size changed")
# warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


def pytest_addoption(parser):
    """Repeat option."""
    parser.addoption('--repeat', action='store', help='Number of times to repeat each test')


def pytest_generate_tests(metafunc):  # pragma: no cover
    # Use --repeat option.
    if metafunc.config.option.repeat is not None:
        count = int(metafunc.config.option.repeat)
        metafunc.fixturenames.append('tmp_ct')
        metafunc.parametrize('tmp_ct', range(count))
