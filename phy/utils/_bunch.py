# -*- coding: utf-8 -*-

"""Bunch class."""

#------------------------------------------------------------------------------
# Bunch class
#------------------------------------------------------------------------------

class Bunch(dict):
    """A dict with additional dot syntax."""
    def __init__(self, *args, **kwargs):
        super(Bunch, self).__init__(*args, **kwargs)
        self.__dict__ = self
