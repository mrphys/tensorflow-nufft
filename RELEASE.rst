Release 0.8.0
=============

This release bumps the supported TensorFlow version to 2.9.0.

Bug Fixes and Other Changes
---------------------------

* Like core TensorFlow, we now compile with `_GLIBCXX_USE_CXX11_ABI=1`.
* Like core TensorFlow, Python wheels now conform to `manylinux2014`, an upgrade
  from `manylinux2010`.
* The op library now links against the static form of the CUDA runtime
  library.
* FINUFFT code is now fully integrated.

Known Caveats
-------------

* The op library does not link against the the static cuFFT library, which will
  result in unresolved symbol errors when used in a system without a CUDA
  installation. This will be addressed in a future release (see also issue #24).
