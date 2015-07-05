# -*- coding: utf-8 -*-

"""Utility functions for test datasets."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import hashlib
import os
import os.path as op

try:
    import requests
    from requests import get
except ImportError:
    get = None

from .logging import debug, info, warn
from .settings import _phy_user_dir, _ensure_dir_exists


#------------------------------------------------------------------------------
# Utility functions
#------------------------------------------------------------------------------

_BASE_URL = {
    'cortexlab': 'http://phy.cortexlab.net/data/samples/',
    'github': 'https://raw.githubusercontent.com/kwikteam/phy-data/master/',
    'local': 'http://localhost:8000/',
}


def md5(path, blocksize=2 ** 20):
    """Compute the checksum of a file."""
    m = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            buf = f.read(blocksize)
            if not buf:
                break
            m.update(buf)
    return m.hexdigest()


def download_file(url, output=None, checksum=None):
    """Download a binary file from an URL."""
    if output is None:
        output = url.split('/')[-1]
    if op.exists(output):
        info("The file {} already exists: skipping.".format(output))
        return
    if not get:
        raise ImportError("Please install the requests package.")
    try:
        r = get(url, stream=True)
    except (requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError,
            ):
        raise RuntimeError("Unable to download `{}`.".format(url))
    if r.status_code != 200:
        warn("Error while downloading `{}`.".format(url))
        r.raise_for_status()
    info("Downloading {0}...".format(url))
    with open(output, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                f.flush()
    if checksum is not None:
        if md5(output) != checksum:
            raise RuntimeError("The checksum of the downloaded file doesn't "
                               "match the provided checksum.")


def _download(url):
    if not get:
        raise ImportError("Please install the requests package.")
    try:
        return get(url).text
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Unable to download `{}`.".format(url))


def _download_test_data(name, phy_user_dir=None, force=False):
    phy_user_dir = phy_user_dir or _phy_user_dir()
    dir = op.join(phy_user_dir, 'test_data')
    _ensure_dir_exists(dir)
    path = op.join(dir, name)
    if not force and op.exists(path):
        return path
    url = _BASE_URL['github'] + 'test/' + name
    download_file(url, output=path)
    return path


def download_sample_data(name, output_dir=None, base='cortexlab'):
    """Download a sample dataset.

    Parameters
    ----------

    name : str
        Name of the sample dataset to download.
    output_dir : str
        The directory where to save the file.
    base : str
        The id of the base URL. Can be `'cortexlab'` or `'github'`.

    """
    if output_dir is None:
        output_dir = '.'
    if not output_dir.endswith('/'):
        output_dir = output_dir + '/'
    output_dir = op.realpath(op.dirname(output_dir))
    if not op.exists(output_dir):
        os.mkdir(output_dir)
    name, ext = op.splitext(name)
    if not ext:
        ext_list = ('.kwik', '.kwx', '.raw.kwd')
    else:
        ext_list = (ext,)
    outputs = []
    for ext in ext_list:
        url = _BASE_URL[base] + name + ext
        output = op.join(output_dir, name + ext)
        url_checksum = _BASE_URL[base] + name + ext + '.md5'
        # Try to download the md5 hash.
        try:
            checksum = _download(url_checksum).split(' ')[0]
        except RuntimeError as e:
            warn("The md5 file could not be found at `{}`.".format(
                 url_checksum))
            debug(e)
            checksum = None
        # Try to download the requested file.
        try:
            download_file(url, output=output, checksum=checksum)
            outputs.append(output)
        except RuntimeError as e:
            warn("The data file could not be found at `{}`.".format(
                 url))
            debug(e)
    if outputs:
        return outputs
