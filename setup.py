from distutils.core import setup


setup(
    name='spycial',
    description='Special functions written in Python and accelerated by Numba',
    version='0.1.1',
    author='Josh Wilson',
    url='https://github.com/person142/spycial',
    packages=['spycial'],
    install_requires=['numba', 'numpy'],
)
