"""Nonuniform fast Fourier transform (NUFFT) for TensorFlow v2."""

import glob
from setuptools import find_packages, setup, Extension
from distutils.sysconfig import get_config_vars
import os

try:
    import tensorflow as tf
except ModuleNotFoundError as err:
    raise ModuleNotFoundError(
        "TensorFlow is required to proceed with the tensorflow_nufft installation."
        ) from err

with open('VERSION') as version_file:
    VERSION = version_file.read().strip()

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f.readlines()]

REQUIRED_PACKAGES = requirements

DOCLINES = __doc__.split('\n')

SOURCES = [

]

base_dir = os.path.dirname(os.path.realpath(__file__))
INCLUDE_DIRS=[
    os.path.join(base_dir, 'third_party/finufft/include')
]

CC_FLAGS = [
    '-fPIC',
    '-O2',
    '-std=c++11'
]

LD_FLAGS = [
    '-shared'
]

SOURCES += glob.glob('cc/ops/*.cc')
SOURCES += glob.glob('cc/kernels/*.cc')
CC_FLAGS += tf.sysconfig.get_compile_flags()
LD_FLAGS += tf.sysconfig.get_link_flags()

tensorflow_nufft_ext = Extension('tensorflow_nufft/libtensorflow_nufft',
                     sources=SOURCES,
                     include_dirs=INCLUDE_DIRS,
                     libraries=['finufft'],
                    #  library_dirs=[os.path.join(base_dir, 'third_party/finufft/lib-static')],
                    #  extra_objects=[os.path.join(base_dir, 'third_party/finufft/lib-static/libfinufft.a')],
                     extra_compile_args=CC_FLAGS,
                     extra_link_args=LD_FLAGS,
                     optional=True)

# Remove platform-specific info from the extension suffix. This environment
# variable is used by `build_ext` to name the library file.
get_config_vars()['EXT_SUFFIX'] = '.so'

setup(
    name='tensorflow_nufft',
    version=VERSION,
    description=DOCLINES[0],
    long_description='\n'.join(DOCLINES[2:]),
    long_description_content_type="text/markdown",
    author='Javier Montalt-Tordera',
    author_email='javier.tordera.17@ucl.ac.uk',
    url='https://github.com/mrphys/tensorflow_nufft',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: GPU',
        'Environment :: GPU :: NVIDIA CUDA',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: C++',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    license="Apache 2.0",
    keywords=['tensorflow', 'nufft'],
    install_requires=REQUIRED_PACKAGES,
    python_requires='>=3.6',
    ext_modules=[tensorflow_nufft_ext]
)
