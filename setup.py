# -*- coding: utf-8 -*-
# flake8: noqa

"""Installation script."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os
import os.path as op
import re

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


#------------------------------------------------------------------------------
# Setup
#------------------------------------------------------------------------------


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


requirements = [
]

test_requirements = [
]

setup(
    name='phy',
    version=version,
    description='Electrophysiological data analysis.',
    long_description=readme,
    author='Kwik Team',
    author_email='cyrille.rossant at gmail.com',
    url='https://github.com/kwikteam/phy',
    packages=_package_tree('phy'),
    package_dir={'phy': 'phy'},
    package_data={
        'phy': ['*.vert', '*.frag', '*.glsl'],
    },
    entry_points={
        'console_scripts': [
            'phy=phy.scripts.phy_script:main',
        ],
    },
    include_package_data=True,
    install_requires=requirements,
    license="BSD",
    zip_safe=False,
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
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],
    tests_require=test_requirements
)
