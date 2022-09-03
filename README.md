# TensorFlow NUFFT

[![PyPI](https://badge.fury.io/py/tensorflow-nufft.svg)](https://badge.fury.io/py/tensorflow-nufft)
[![Build](https://github.com/mrphys/tensorflow-nufft/actions/workflows/build-package.yml/badge.svg)](https://github.com/mrphys/tensorflow-nufft/actions/workflows/build-package.yml)
[![Docs](https://img.shields.io/badge/api-reference-blue.svg)](https://mrphys.github.io/tensorflow-nufft)
[![DOI](https://zenodo.org/badge/382718757.svg)](https://zenodo.org/badge/latestdoi/382718757)

<!-- start-intro -->

TensorFlow NUFFT is a fast, native non-uniform fast Fourier transform op for
TensorFlow. It provides:

- Fast CPU/GPU kernels. The TensorFlow framework automatically handles device
  placement as usual.
- A simple, well-documented Python interface.
- Gradient definitions for automatic differentiation.
- Shape functions to support static shape inference.

The underlying algorithm is based on the NUFFT implementation by the Flatiron
Institute. Please refer to
[FINUFFT](https://github.com/flatironinstitute/finufft/) and
[cuFINUFFT](https://github.com/flatironinstitute/cufinufft/) for more details.

<!-- end-intro -->

<!-- start-install -->

## Installation

You can install TensorFlow NUFFT with ``pip``:

```
pip install tensorflow-nufft
```

Note that only Linux wheels are currently being provided.

### TensorFlow compatibility

Each TensorFlow NUFFT release is compiled against a specific version of
TensorFlow. To ensure compatibility, it is recommended to install matching
versions of TensorFlow and TensorFlow NUFFT according to the table below.

| TensorFlow NUFFT Version | TensorFlow Compatibility | Release Date |
| ------------------------ | ------------------------ | ------------ |
| v0.9.0                   | v2.10.x                  |              |
| v0.8.1                   | v2.9.x                   | Jun 23, 2022 |
| v0.8.0                   | v2.9.x                   | May 20, 2022 |
| v0.7.3                   | v2.8.x                   | May 4, 2022  |
| v0.7.2                   | v2.8.x                   | Apr 29, 2022 |
| v0.7.1                   | v2.8.x                   | Apr 6, 2022  |
| v0.7.0                   | v2.8.x                   | Feb 8, 2022  |
| v0.6.0                   | v2.7.x                   | Jan 27, 2022 |
| v0.5.0                   | v2.7.x                   | Dec 12, 2021 |
| v0.4.0                   | v2.7.x                   | Nov 8, 2021  |
| v0.3.2                   | v2.6.x                   | Aug 18, 2021 |
| v0.3.1                   | v2.6.x                   | Aug 18, 2021 |
| v0.3.0                   | v2.6.x                   | Aug 13, 2021 |

<!-- end-install -->

## Documentation

Visit the [docs](https://mrphys.github.io/tensorflow-nufft/) for the API
reference and examples of usage.

## Issues

If you use this package and something does not work as you expected, please
[file an issue](https://github.com/mrphys/tensorflow-nufft/issues/new)
describing your problem. We're here to help!

## Credits

If you find this software useful in your research, please
[cite us](https://doi.org/10.5281/zenodo.5198288).

## Contributors

Thanks to all our contributors: `jmontalt <https://github.com/jmontalt>`_,
`chaithyagr <https://github.com/chaithyagr>`_.

Contributions of any kind are welcome!
