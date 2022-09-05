# Release 0.9.0


## Major Features and Improvements

- Added new class `Options` to be passed to `nufft` to specify advanced options.
- Added new argument `options` to `nufft` to allow specifying advanced options.


## Bug Fixes and Other Changes

- FFTW planning rigor now defaults to `MEASURE` instead of `ESTIMATE`, but this
  can now be changed via the `Options` mechanism.
