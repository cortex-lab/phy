# -*- coding: utf-8 -*-
# flake8: noqa

"""Installation script."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import re
import os.path as op

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


#------------------------------------------------------------------------------
# Setup
#------------------------------------------------------------------------------

readme = open('README.md').read()


# Find version number from `__init__.py` without executing it.
filename = op.join(op.dirname(op.realpath(__file__)), 'phy/__init__.py')
with open(filename, 'r') as f:
    version = re.search(r"__version__ = '([^']+)'", f.read()).group(1)


requirements = [
    # TODO: put package requirements here
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='phy',
    version=version,
    description='Electrophysiological data analysis.',
    long_description=readme,
    author='Kwik Team',
    author_email='cyrille.rossant at gmail.com',
    url='https://github.com/kwikteam/phy',
    packages=[
        'phy',
    ],
    package_dir={'phy': 'phy'},
    entry_points={
        'console_scripts': [
            'klustaviewa=phy.scripts.klustaviewa:main',
        ],
    },
    include_package_data=True,
    install_requires=requirements,
    license="BSD",
    zip_safe=False,
    keywords='phy,data analysis,electrophysiology,neuroscience',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],
    tests_require=test_requirements
)
