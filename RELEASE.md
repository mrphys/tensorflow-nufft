# Release 0.11.0

## Bug Fixes and Other Changes

- Improved error reporting for invalid `grid_shape` arguments. `nufft` will
  now raise an informative error when `grid_shape` has an invalid length or
  when the user fails to provide it for type-1 transforms. Previously, `nufft`
  would have behaved erratically or crashed.
