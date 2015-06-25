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

from ..datasets import (_download,
                        _download_test_data,
                        download_file,
                        download_sample_data,
                        _BASE_URL,
                        )
from ..tempdir import TemporaryDirectory


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

# Test URL and data
_URL = 'http://test/data'
_DATA = np.linspace(0., 1., 100000).astype(np.float32)
_CHECKSUM = '7d257d0ae7e3af8ca3574ccc3a4bf072'


def _add_mock_response(url, body, file_type='binary'):
    content_type = ('application/octet-stream'
                    if file_type == 'binary' else 'text/plain')
    responses.add(responses.GET, url,
                  body=body,
                  status=200,
                  content_type=content_type,
                  )


def setup():
    _add_mock_response(_URL, _DATA.tostring())
    _add_mock_response(_URL + '.md5', _CHECKSUM)


def teardown():
    responses.reset()


#------------------------------------------------------------------------------
# Test utility functions
#------------------------------------------------------------------------------

@responses.activate
def test_download_error():
    with TemporaryDirectory() as tmpdir:
        path = op.join(tmpdir, 'test')
        with raises(Exception):
            download_file(_URL + '_notfound', path)


@responses.activate
def test_download_checksum():
    assert _download(_URL + '.md5') == _CHECKSUM


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


@responses.activate
def test_download_sample_data():
    name = 'sample'
    url = _BASE_URL['cortexlab'] + name
    for ext in ('.kwik', '.kwx', '.raw.kwd'):
        _add_mock_response(url + ext, _DATA.tostring())
        _add_mock_response(url + ext + '.md5', _CHECKSUM)

    with TemporaryDirectory() as tmpdir:
        output_dir = op.join(tmpdir, name)
        download_sample_data(name, output_dir)
        for ext in ('.kwik', '.kwx', '.raw.kwd'):
            with open(op.join(output_dir, name + ext), 'rb') as f:
                data = f.read()
            ae(np.fromstring(data, np.float32), _DATA)


@responses.activate
def test_dat_file():
    data = np.random.randint(size=(20000, 4),
                             low=-100, high=100).astype(np.int16)
    fn = 'test-4ch-1s.dat'
    _add_mock_response(_BASE_URL['github'] + 'test/' + fn,
                       data.tostring())
    with TemporaryDirectory() as tmpdir:
        path = _download_test_data(fn, tmpdir)
        with open(path, 'rb') as f:
            arr = np.fromfile(f, dtype=np.int16).reshape((-1, 4))
        assert arr.shape == (20000, 4)
