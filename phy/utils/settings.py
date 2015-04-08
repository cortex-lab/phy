# -*- coding: utf-8 -*-

import os.path

class Settings(dict):
    """
    use like or dict or an object
    SETTINGS.KWIK_DIRS
    SETTINGS.get('KWIK_DIRS')

    """
    def __init__(self, *args, **kwargs):
        super(Settings, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def load(self, filename):
        """ load a python file """
        settings = {}
        execfile(os.path.expanduser(filename), {}, settings)

        self.update(settings)
        print "data:", self

def init():
    s = settings()
    s.load("~/.phy/settings.py")

SETTINGS = Settings()

init()
