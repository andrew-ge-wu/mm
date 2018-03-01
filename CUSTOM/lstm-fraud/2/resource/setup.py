import os
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

here = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(here, 'README.md')).read()

setup(
    name='pipeline',
    version='0.0.1',
    description='',
    long_description=README,
    py_modules=['loss'],
    packages=['pipeline', 'iterators'],
)
