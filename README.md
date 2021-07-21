# tensorflow_nufft

This is a TensorFlow op implementing the non-uniform fast Fourier transform
(NUFFT):

 - Native C++/CUDA kernels for CPU/GPU.
 - Python/TensorFlow interface.
 - Automatic differentiation.
 - Automatic shape inference.

The core NUFFT implementation is that of the Flatiron Institute. Please see the
original [FINUFFT]() and [cuFINUFFT]() repositories for details. The main
contribution of this package is the TensorFlow functionality.

## Installation

### Install

The easiest way to install `tensorflow-nufft` is via pip.

```
pip install tensorflow-nufft
```

Note that only Linux binaries are currently being provided.

### Contributions
All contributions are welcome.


### Example
For usage examples, see [examples](examples). 


### Citations

