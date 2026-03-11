"""py.test utilities."""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

import os
import logging
import warnings
from functools import wraps

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
# Keep Qt tests headless by default so GUI windows do not interrupt the desktop.
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
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
warnings.filterwarnings(
    'ignore',
    category=DeprecationWarning,
    module=r'OpenGL\.GL\.VERSION\.GL_2_0',
)


def _suppress_pyopengl_tostring_warning():
    try:
        from OpenGL.GL.VERSION import GL_2_0
    except Exception:
        return

    def _wrap(func):
        @wraps(func)
        def inner(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore',
                    message=r'tostring\(\) is deprecated\. Use tobytes\(\) instead\.',
                    category=DeprecationWarning,
                )
                return func(*args, **kwargs)

        return inner

    for name in ('glGetActiveAttrib', 'glGetActiveUniform'):
        func = getattr(GL_2_0, name, None)
        if callable(func):
            setattr(GL_2_0, name, _wrap(func))


_suppress_pyopengl_tostring_warning()

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


def pytest_collection_modifyitems(session, config, items):
    """Run app tests after the rest of the suite."""
    items.sort(key=lambda item: ('/phy/apps/' in str(item.fspath).replace(os.sep, '/'), str(item.fspath)))
