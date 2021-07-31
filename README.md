# TensorFlow NUFFT

[![PyPI](https://badge.fury.io/py/tensorflow-nufft.svg)](https://badge.fury.io/py/tensorflow-nufft)
[![build](https://github.com/mrphys/tensorflow-nufft/actions/workflows/build-package.yml/badge.svg)](https://github.com/mrphys/tensorflow-nufft/actions/workflows/build-package.yml)

This is an implementation of the non-uniform fast Fourier transform (NUFFT) in
TensorFlow. The op provides:

 - Native C++/CUDA kernels for CPU/GPU.
 - Python/TensorFlow interface.
 - Automatic differentiation.
 - Automatic shape inference.

The core NUFFT implementation is that of the Flatiron Institute. Please see the
original [FINUFFT](https://github.com/flatironinstitute/finufft) and
[cuFINUFFT](https://github.com/flatironinstitute/cufinufft) repositories for
details. The main contribution of this package is the TensorFlow op wrapper and
functionality.

## Installation

The easiest way to install `tensorflow-nufft` is via pip.

```
pip install tensorflow-nufft
```

Note that only Linux binaries are currently being provided.

## Example
For usage examples, see [examples](tools/examples). 

## Cite this


## Contributions
All contributions are welcome.
