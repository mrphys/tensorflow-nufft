# Release 0.11.0

## Major Features and Improvements

- Added new option `points_range` to control the range supported by the
  algorithm. This option provides a trade-off between flexibility and
  performance.
- Added new option `debugging.check_points_range` to assert that the input
  points lie within the supported range.

## Bug Fixes and Other Changes

- `nufft` type-1 will now raise an error when `source` and `points` have
  incompatible samples dimensions. Previously the computation would have
  proceeded, ignoring any additional samples in `source`.
- Improved error reporting for invalid `grid_shape` arguments. `nufft` will
  now raise an informative error when `grid_shape` has an invalid length or
  when the user fails to provide it for type-1 transforms. Previously, `nufft`
  would have behaved erratically or crashed.
