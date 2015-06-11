# -*- coding: utf-8 -*-
from __future__ import print_function

"""Automatic release tools."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import sys
import os
import os.path as op
import re
import six


# -----------------------------------------------------------------------------
# Messing with version in __init__.py
# -----------------------------------------------------------------------------

root = op.realpath(op.join(op.dirname(__file__), '../'))
_version_pattern = r"__version__ = '([0-9\.]+)((?:\.dev)?)([0-9]+)'"
_version_replace = r"__version__ = '{}{}{}'"


def _path(fn):
    return op.realpath(op.join(root, fn))


def _update_version(dev_n='+1', dev=True):
    fn = _path('phy/__init__.py')
    dev = '.dev' if dev else ''

    def func(m):
        if dev:
            if isinstance(dev_n, six.string_types):
                n = int(m.group(3)) + int(dev_n)
            assert n >= 0
        else:
            n = ''
        if not m.group(2):
            raise ValueError()
        return _version_replace.format(m.group(1), dev, n)

    with open(fn, 'r') as f:
        contents = f.read()

    contents_new = re.sub(_version_pattern, func, contents)

    with open(fn, 'w') as f:
        f.write(contents_new)


def _increment_dev_version():
    _update_version('+1')


def _decrement_dev_version():
    _update_version('-1')


def _set_final_version():
    _update_version(dev=False)


# -----------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------

def _build_package():
    pass


def _upload_pypi():
    pass


def _git_commit(message, push=False):
    pass


def _create_gh_release():
    pass


def _build_docker():
    pass


def _test_docker():
    pass
