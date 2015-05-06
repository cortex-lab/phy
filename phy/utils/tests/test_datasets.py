# -*- coding: utf-8 -*-

"""Tests of dataset utility functions."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op

import numpy as np
from numpy.testing import assert_array_equal as ae
import responses
from pytest import raises

from ..datasets import download_file, md5
from ..tempdir import TemporaryDirectory


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

# Test URL and data
_URL = 'http://test/data'
_DATA = np.linspace(0., 1., 100000).astype(np.float32)
_CHECKSUM = '7d257d0ae7e3af8ca3574ccc3a4bf072'


def setup():
    responses.add(responses.GET,
                  _URL,
                  body=_DATA.tostring(),
                  status=200,
                  content_type='application/octet-stream',
                  )


def teardown():
    responses.reset()


#------------------------------------------------------------------------------
# Test utility functions
#------------------------------------------------------------------------------

def _test_download_file(checksum=None):
    with TemporaryDirectory() as tmpdir:
        path = op.join(tmpdir, 'test.kwik')
        download_file(_URL, path, checksum=checksum)
        with open(path, 'rb') as f:
            data = f.read()
    ae(np.fromstring(data, np.float32), _DATA)


@responses.activate
def test_download_no_checksum():
    _test_download_file(checksum=None)


@responses.activate
def test_download_valid_checksum():
    _test_download_file(checksum=_CHECKSUM)


@responses.activate
def test_download_invalid_checksum():
    with raises(RuntimeError):
        _test_download_file(checksum=_CHECKSUM[:-1])
