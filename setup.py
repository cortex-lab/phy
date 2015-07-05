# -*- coding: utf-8 -*-
# flake8: noqa

"""Installation script."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os
import os.path as op
import sys
import re

from setuptools import setup
from setuptools.command.test import test as TestCommand


#------------------------------------------------------------------------------
# Setup
#------------------------------------------------------------------------------

class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ['-s', '--cov-report', 'term-missing',
                            '--cov', 'phy']

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        #import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


def _package_tree(pkgroot):
    path = op.dirname(__file__)
    subdirs = [op.relpath(i[0], path).replace(op.sep, '.')
               for i in os.walk(op.join(path, pkgroot))
               if '__init__.py' in i[2]]
    return subdirs


curdir = op.dirname(op.realpath(__file__))
readme = open(op.join(curdir, 'README.md')).read()


# Find version number from `__init__.py` without executing it.
filename = op.join(curdir, 'phy/__init__.py')
with open(filename, 'r') as f:
    version = re.search(r"__version__ = '([^']+)'", f.read()).group(1)


setup(
    name='phy',
    version=version,
    license="BSD",
    description='Spike sorting and ephys data analysis '
                'for 1000 channels and beyond',
    long_description=readme,
    author='Kwik Team',
    author_email='cyrille.rossant at gmail.com',
    url='https://phy.cortexlab.net',
    packages=_package_tree('phy'),
    package_dir={'phy': 'phy'},
    package_data={
        'phy': ['*.vert', '*.frag', '*.glsl', '*.html', '*.css', '*.prb'],
    },
    entry_points={
        'console_scripts': [
            'phy=phy.scripts.phy_script:main',
        ],
    },
    include_package_data=True,
    # zip_safe=False,
    keywords='phy,data analysis,electrophysiology,neuroscience',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        "Framework :: IPython",
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
    ],
    cmdclass={'test': PyTest},
)
