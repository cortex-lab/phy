# -*- coding: utf-8 -*-
from .utils import default_logger

__author__ = 'Kwik Team'
__email__ = 'cyrille.rossant at gmail.com'
__version__ = '0.1.0-alpha'

# Set up the default logger.
default_logger()


def prepare_js():
    """ This is needed to map js/css to the nbextensions folder
    """
    from IPython.html import nbextensions
    import os
    pkgdir = os.path.dirname(__file__)
    nbextensions.install_nbextension(pkgdir, symlink=True, user=True, destination='phy')

#prepare_js()
