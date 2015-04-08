# -*- coding: utf-8 -*-

"""Test settings."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import raises

from ..settings import _Settings


#------------------------------------------------------------------------------
# Test settings
#------------------------------------------------------------------------------

def test_settings():
    s = _Settings()

    # Namespaces are mandatory.
    with raises(ValueError):
        s._get('a')

    # None is returned if a key doesn't exist.
    assert s._get('test.a') is None

    s._set({'test.a': 3})
    assert s._get('test.a') == 3
