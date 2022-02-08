TensorFlow NUFFT
================

|pypi| |build| |docs| |doi|

.. |pypi| image:: https://badge.fury.io/py/tensorflow-nufft.svg
    :target: https://badge.fury.io/py/tensorflow-nufft
.. |build| image:: https://github.com/mrphys/tensorflow-nufft/actions/workflows/build-package.yml/badge.svg
    :target: https://github.com/mrphys/tensorflow-nufft/actions/workflows/build-package.yml
.. |docs| image:: https://img.shields.io/badge/api-reference-blue.svg
    :target: https://mrphys.github.io/tensorflow-nufft
.. |doi| image:: https://zenodo.org/badge/382718757.svg
    :target: https://zenodo.org/badge/latestdoi/382718757

.. start-intro

TensorFlow NUFFT is a fast, native non-uniform fast Fourier transform op for
TensorFlow. It provides:

* Fast CPU/GPU kernels. The TensorFlow framework automatically handles device
  placement as usual.
* A simple, well-documented Python interface.
* Gradient definitions for automatic differentiation.
* Shape functions to support static shape inference.

The underlying algorithm is based on the NUFFT implementation by the Flatiron
Institute. Please refer to
`FINUFFT <https://github.com/flatironinstitute/finufft/>`_ and
`cuFINUFFT <https://github.com/flatironinstitute/cufinufft/>`_ for
more details.

.. end-intro

Installation
------------

.. start-install

You can install TensorFlow NUFFT with ``pip``:

.. code-block:: console

    $ pip install tensorflow-nufft

Note that only Linux wheels are currently being provided.

TensorFlow Compatibility
^^^^^^^^^^^^^^^^^^^^^^^^

Each TensorFlow NUFFT release is compiled against a specific version of
TensorFlow. To ensure compatibility, it is recommended to install matching
versions of TensorFlow and TensorFlow NUFFT according to the table below.

========================  ========================  ============
TensorFlow NUFFT Version  TensorFlow Compatibility  Release Date
========================  ========================  ============
v0.7.0                    v2.8.x                    Feb 8, 2022
v0.6.0                    v2.7.x                    Jan 27, 2022
v0.5.0                    v2.7.x                    Dec 12, 2021
v0.4.0                    v2.7.x                    Nov 8, 2021
v0.3.2                    v2.6.x                    Aug 18, 2021
v0.3.1                    v2.6.x                    Aug 18, 2021
v0.3.0                    v2.6.x                    Aug 13, 2021
========================  ========================  ============

.. end-install

Documentation
-------------

Visit the `docs <https://mrphys.github.io/tensorflow-nufft/>`_ for the API
reference and examples of usage. 

Contributions
-------------

If you use this package and something does not work as you expected, please
`file an issue <https://github.com/mrphys/tensorflow-nufft/issues/new>`_
describing your problem. We will do our best to help.

Contributions are very welcome. Please create a pull request if you would like
to make a contribution.

Citation
--------

If you find this software useful in your work, please
`cite us <https://doi.org/10.5281/zenodo.5198288>`_.
