"""TensorFlow NUFFT.

Implementation of the non-uniform Fourier transform in TensorFlow.
"""

from setuptools import find_packages
from setuptools import setup
from setuptools.dist import Distribution
from setuptools.command.install import install as _install

PROJECT_NAME = 'tensorflow-nufft'

with open('VERSION') as version_file:
    VERSION = version_file.read().strip()

with open("requirements.txt") as f:
    REQUIRED_PACKAGES = [line.strip() for line in f.readlines()]

DOCLINES = __doc__.split('\n')

class install(_install):

    def finalize_options(self):
        _install.finalize_options(self)
        self.install_lib = self.install_platlib

class BinaryDistribution(Distribution):

    def has_ext_modules(self):
        return True
    
    def is_pure(self):
        return False

setup(
    name=PROJECT_NAME,
    version=VERSION,
    description=DOCLINES[0],
    long_description='\n'.join(DOCLINES[2:]),
    long_description_content_type="text/markdown",
    author='Javier Montalt-Tordera',
    author_email='javier.tordera.17@ucl.ac.uk',
    url='https://github.com/mrphys/tensorflow_nufft',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: GPU',
        'Environment :: GPU :: NVIDIA CUDA',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: C++',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    distclass=BinaryDistribution,
    license="Apache 2.0",
    keywords=['tensorflow', 'nufft'],
    cmdclass={'install': install},
    include_package_data=True,
    zip_safe=False,
    install_requires=REQUIRED_PACKAGES
)
