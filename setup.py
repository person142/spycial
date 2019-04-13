from setuptools import setup
from os import path


def get_long_description():
    this_directory = path.abspath(path.dirname(__file__))
    with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
    return long_description


setup(
    name='spycial',
    description='Special functions written in Python and accelerated by Numba',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    version='0.2',
    author='Josh Wilson',
    url='https://github.com/person142/spycial',
    packages=['spycial'],
    install_requires=['numba', 'numpy'],
)
