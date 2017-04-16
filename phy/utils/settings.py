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

SETTINGS = Settings()

def init():
    SETTINGS.load("~/.phy/settings.py")

init()

def list_kwik():
    ret = []
    for d in SETTINGS.KWIK_DIRS:
        for root, dirs, files in os.walk(os.path.expanduser(d)):
            for f in files:
                if f.endswith(".kwik"):
                    ret.append(os.path.join(root, f))
    return ret
