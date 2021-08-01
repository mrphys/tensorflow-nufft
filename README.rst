TensorFlow NUFFT
================

|pypi| |build| |docs|

.. |pypi| image:: https://badge.fury.io/py/tensorflow-nufft.svg
    :target: https://badge.fury.io/py/tensorflow-nufft
.. |build| image:: https://github.com/mrphys/tensorflow-nufft/actions/workflows/build-package.yml/badge.svg
    :target: https://github.com/mrphys/tensorflow-nufft/actions/workflows/build-package.yml
.. |docs| image:: https://img.shields.io/badge/api-reference-blue.svg
    :target: https://mrphys.github.io/tensorflow-nufft

.. start-intro

TensorFlow NUFFT is a fast, native non-uniform fast Fourier transform op for
TensorFlow. It provides:

* Fast CPU/GPU kernels. The TensorFlow framework automatically handles device
  placement as usual.
* A simple, well-documented Python interface.
* Gradient definitions for automatic differentiation.
* Shape functions to support static shape inference.

The underlying NUFFT implementation is that of the Flatiron Institute. Please
refer to the `FINUFFT <https://github.com/flatironinstitute/finufft/>`_ and
`cuFINUFFT <https://github.com/flatironinstitute/cufinufft/>`_ repositories for
details.

.. end-intro

Installation
------------

.. start-install

You can install TensorFlow NUFFT with ``pip``:

.. code-block:: console

    $ pip install tensorflow-nufft

Note that only Linux wheels are currently being provided.

.. end-install
