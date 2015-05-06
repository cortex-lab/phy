# -*- coding: utf-8 -*-

"""Tests of dataset utility functions."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os.path as op

import numpy as np
from numpy.testing import assert_array_equal as ae
import responses

from ..datasets import download_file
from ..tempdir import TemporaryDirectory


#------------------------------------------------------------------------------
# Test utility functions
#------------------------------------------------------------------------------

_URL = 'http://test/data'


_DATA = np.random.rand(100000).astype(np.float32)


def setup():
    responses.add(responses.GET,
                  _URL,
                  body=_DATA.tostring(),
                  status=200,
                  content_type='application/octet-stream',
                  )


def teardown():
    responses.reset()


@responses.activate
def test_download_file():
    with TemporaryDirectory() as tmpdir:
        path = op.join(tmpdir, 'test.kwik')
        download_file(_URL, path)
        with open(path, 'rb') as f:
            data = f.read()
    ae(np.fromstring(data, np.float32), _DATA)
