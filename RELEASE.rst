Release 0.8.1
=============

Bug Fixes and Other Changes
---------------------------

* We now use TensorFlow's stream executor to perform FFTs.
* This library should now be fully usable in systems without a CUDA
  installation (CPU only).
