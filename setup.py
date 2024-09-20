#!/usr/bin/env python

import glob
from setuptools import setup, find_packages

# Get metadata from setup.cfg (optional) or directly include here
PACKAGENAME = 'FITS_tools'
DESCRIPTION = 'Tools for manipulating FITS images using primarily scipy & native python routines'
AUTHOR = 'Adam Ginsburg'
AUTHOR_EMAIL = 'adam.g.ginsburg@gmail.com'
LICENSE = 'BSD'
URL = 'http://github.com/keflavich/FITS_tools/'
VERSION = '0.3.1.dev'

# Modernized setup
setup(
    name=PACKAGENAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=open('README.rst').read(),
    long_description_content_type='text/x-rst',  # Ensure correct content type
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    url=URL,
    packages=find_packages(),  # Automatically find all packages
    scripts=glob.glob('scripts/*'),  # Any scripts you want to include
    install_requires=[
        'astropy>=4.0',
        'numpy',
        'scipy',
        'matplotlib',
        'spectral_cube'
    ],  # Specify dependencies
    classifiers=[  # Add more classifiers to categorize your package
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    python_requires='>=3.7',  # Minimum Python version
    zip_safe=False,
)
