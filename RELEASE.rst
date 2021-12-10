Release 0.5.0
=============

<INSERT SMALL BLURB ABOUT RELEASE FOCUS AREA AND POTENTIAL TOOLCHAIN CHANGES>

Breaking Changes
----------------

This release does not contain known breaking changes.

Known Caveats
-------------

* A major refactoring of the code has begun in order to improve the integration
  of FINUFFT with the TensorFlow framework. This process is not complete yet and
  will continue on future releases.

Major Features and Improvements
-------------------------------

* `nufft` and `spread` will now accept tensors for the `grid_shape` input. This
  improves flexibility and means that the grid shape no longer needs to be known
  at graph creation time.

Bug Fixes and Other Changes
---------------------------

* Fixed a bug on the CPU kernel that would cause a segmentation fault when
  running with inter-op parallelism. The issue was caused by the FFTW planner,
  which is not thread-safe, being accessed concurrently by multiple op
  instances. Access to the FFTW planner is now protected in a critical section. 
* `nufft`, `interp` and `spread` will now honour the TensorFlow framework's
  intra-op parallelism setting. From now on, the number of intra-op threads used
  by the CPU kernel can be specified by the user with
  `tf.config.threading.set_intra_op_parallelism_threads`.
* GPU kernel launches, memory allocations and data copies will now be enqueued
  to the correct devices and streams as requested by the TensorFlow framework.
