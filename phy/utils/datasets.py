# -*- coding: utf-8 -*-

"""Utility functions for test datasets."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import requests


#------------------------------------------------------------------------------
# Utility functions
#------------------------------------------------------------------------------

def download_file(url, output=None):
    """Download a binary file from an URL."""
    if output is None:
        output = url.split('/')[-1]
    r = requests.get(url, stream=True)
    with open(output, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                f.flush()
