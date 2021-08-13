# Copyright 2021 University College London. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Non-uniform fast Fourier transform (NUFFT).

This module contains ops to calculate the NUFFT and some related functionality.
"""

import tensorflow as tf


_nufft_ops = tf.load_op_library(
  tf.compat.v1.resource_loader.get_path_to_datafile('_nufft_ops.so'))


nufft = _nufft_ops.nufft
interp = _nufft_ops.interp
spread = _nufft_ops.spread


@tf.RegisterGradient("NUFFT")
def _nufft_grad(op, grad):
  """Gradients for `nufft`.

  Args:
    op: The `nufft` `tf.Operation`.
    grad: Gradient with respect to the output of the `nufft` op.

  Returns:
    Gradients with respect to the inputs of `nufft`.
  """
  # Get inputs.
  source = op.inputs[0]
  points = op.inputs[1]
  transform_type = op.get_attr('transform_type')
  fft_direction = op.get_attr('fft_direction')
  tol = op.get_attr('tol')
  rank = points.shape[-1]

  # Gradient of type-1 transform is computed using type-2 transform and
  # viceversa.
  if transform_type == b'type_1':   # nonuniform to uniform
    grad_transform_type = 'type_2'  # uniform to nonuniform
    grad_grid_shape = []
  elif transform_type == b'type_2':
    grad_transform_type = 'type_1'
    grad_grid_shape = source.shape[-rank:]

  # Gradient of forward transform is computed using backward transform and
  # viceversa.
  if fft_direction == b'backward':
    grad_fft_direction = 'forward'
  elif fft_direction == b'forward':
    grad_fft_direction = 'backward'

  # Compute the gradients with respect to the `source` tensor.
  grad_source = nufft(grad,
                      points,
                      grid_shape=grad_grid_shape,
                      transform_type=grad_transform_type,
                      fft_direction=grad_fft_direction,
                      tol=tol)

  # Handle broadcasting.
  baxis = -1 if transform_type == b'type_1' else -rank
  src_bshape = tf.shape(source)[:baxis]
  pts_bshape = tf.shape(points)[:-2]
  src_rax, _ = tf.raw_ops.BroadcastGradientArgs(s0=src_bshape,
                            s1=pts_bshape)
  grad_source = tf.reshape(
    tf.math.reduce_sum(grad_source, src_rax), tf.shape(source))

  # Gradients with respect to the points tensor are not implemented.
  return [grad_source, None]


def nudft(source,
          points,
          grid_shape=None,
          transform_type='type_2',
          fft_direction='forward'):
  """Compute the non-uniform discrete Fourier transform.

  .. warning::
    This function explicitly creates a dense DFT matrix and is very
    computationally expensive. In most cases, `tfft.nufft` should be used
    instead. This function exists primarily for testing purposes.

  For the parameters, see `tfft.nufft`.
  """
  def _nudft(inputs):
    src, pts = inputs
    shape = src.shape if transform_type == 'type_2' else grid_shape
    nudft_matrix = _nudft_matrix(
      pts, shape, fft_direction=fft_direction)
    if transform_type == 'type_1':
      nudft_matrix = tf.transpose(nudft_matrix)
    src_vec = tf.reshape(src, [-1])
    return tf.linalg.matvec(nudft_matrix, src_vec)

  # Validate inputs. This also broadcasts `source` and `points` to equal
  # batch shapes.
  source, points, transform_type, fft_direction, grid_shape = \
    _validate_nudft_inputs(
      source, points, transform_type, fft_direction, grid_shape)

  # Flatten batch dimensions.
  rank = points.shape[-1]
  num_points = points.shape[-2]
  batch_shape = points.shape[:-2]

  # Source and target shapes without batch dimensions.
  if transform_type == 'type_1':
    source_shape = source.shape[-1:].as_list()
    batch_shape = source.shape[:-1].as_list()
    target_shape = grid_shape
  elif transform_type == 'type_2':
    source_shape = source.shape[-rank:].as_list()
    batch_shape = source.shape[:-rank].as_list()
    target_shape = [num_points]
  points_shape = points.shape[-2:].as_list()

  source = tf.reshape(source, [-1] + source_shape)
  points = tf.reshape(points, [-1] + points_shape)

  # Apply op for each element in batch.
  target = tf.map_fn(_nudft, [source, points], fn_output_signature=source.dtype)

  # Restore batch dimensions.
  target = tf.reshape(target, batch_shape + target_shape)

  return target


def _nudft_matrix(points, grid_shape, fft_direction):
  """Compute the nonuniform Fourier transform matrix.

  Args:
    points: Nonuniform points, in the range [-pi, pi].
    grid_shape: Shape of the grid.
    fft_direction: Sign of the imaginary unit in the exponential. Must be either
      'backward' or 'forward'.

  Returns:
    The non-uniform Fourier transform matrix.
  """
  rank = len(grid_shape)
  # Compute a grid of frequencies.
  r_vec = [tf.linspace(-size / 2, size / 2 - 1, size) for size in grid_shape]

  r_grid = tf.cast(tf.reshape(
    tf.meshgrid(*r_vec, indexing='ij'),
    [rank, tf.reduce_prod(grid_shape)]), points.dtype)

  points_grid = tf.cast(tf.matmul(
    points, r_grid), _complex_dtype(points.dtype))

  if fft_direction == 'backward':
    nudft_matrix = tf.exp(1j * points_grid)
  elif fft_direction == 'forward':
    nudft_matrix = tf.exp(-1j * points_grid)

#   TODO: scaling?
#   nudft_matrix = nudft_matrix / (
#     np.sqrt(tf.reduce_prod(grid_shape)) * np.power(np.sqrt(2), rank))
  return nudft_matrix


def _validate_nudft_inputs(source,
                           points,
                           transform_type,
                           fft_direction,
                           grid_shape=None,
                           expected_rank=None,
                           expected_grid_shape=None,
                           expected_dtype=None):
  """Validate inputs for non-uniform discrete Fourier transform.

  Args:
    source: The source tensor.
    points: The points tensor.
    transform_type: The type of the transform.
    fft_direction: The sign of the imaginary unit.
    grid_shape: The grid shape.
    expected_rank: The expected rank.
    expected_grid_shape: The expected image shape.
    expected_dtype: The expected dtype.

  Returns:
    Valid `source` and `points` tensors.

  Raises:
    ValueError: If `expected_rank` or `expected_grid_shape` are provided
      and the last dimension of `points` does not match the expected rank.
    TypeError: If `expected_dtype` is provided and the dtype of `source` does
      not match the expected dtype, or if the dtype of `points` does not
      match the real part of the expected dtype.
    TypeError: If the dtype of `points` does not match the real part of the
      dtype of `source`.
    ValueError: If `expected_grid_shape` is provided and the spatial
      dimensions of `source` do not match the expected image shape.
    ValueError: If the batch shapes of `source` and `points` are not
      broadcastable.
  """
  # Check flags.
  transform_type = _validate_enum(
    transform_type, {'type_1', 'type_2'}, 'transform_type')
  fft_direction = _validate_enum(
    fft_direction, {'backward', 'forward'}, 'fft_direction')

  # Check rank.
  rank = points.shape[-1]

  if transform_type == 'type_1':

    if not len(grid_shape) == rank:
      raise ValueError((
        "Invalid `grid_shape` argument: must represent a rank-{} "
        "shape. Received: {}").format(rank, grid_shape))

  if expected_rank:

    if not rank == expected_rank:
      raise ValueError((
        "Invalid shape for `points` argument: "
        "last dimension must be equal to expected rank, which is {}. "
        "Received: {}").format(expected_rank, rank))

  # Check that dtype for `source` matches the expected dtype.
  if expected_dtype:

    if not source.dtype == expected_dtype:
      raise TypeError((
        "Invalid dtype for `source` argument: "
        "must match the expected dtype, which is {}. "
        "Received: {}").format(expected_dtype, source.dtype))

  expected_dtype = source.dtype

  # Check that dtype for `points` matches the expected dtype.
  if not points.dtype == expected_dtype.real_dtype:
    raise TypeError((
      "Invalid dtype for `points` argument: "
      "must match the real part of the expected dtype, which is {}. "
      "Received: {}").format(expected_dtype.real_dtype, points.dtype))

  # Check that spatial dimensions of input `source` match the expected modes
  # shape.
  if expected_grid_shape:

    if transform_type == 'type_1':
      if not grid_shape == expected_grid_shape:
        raise ValueError((
          "Invalid `grid_shape` argument: "
          "expected {}. Received: {}").format(
            expected_grid_shape, grid_shape))

    if transform_type == 'type_2':
      if not source.shape[-rank:] == expected_grid_shape:
        raise ValueError((
          "Invalid shape for `source` argument: "
          "the modes shape (i.e., dimensions {}) must match "
          "the expected modes shape, which is {}. Received: {}").format(
            tuple(range(-rank, 0)), expected_grid_shape, source.shape))

  # Check that batch shapes for `source` and `points` are broadcastable, and
  # broadcast them to a common shape.
  if transform_type == 'type_1':
    source_shape = source.shape[-1:] # Without batch dimension.
    batch_shape_source = source.shape[:-1]
  elif transform_type == 'type_2':
    source_shape = source.shape[-rank:] # Without batch dimension.
    batch_shape_source = source.shape[:-rank]

  points_shape = points.shape[-2:] # Without batch dimension.
  batch_shape_points = points.shape[:-2]

  try:
    batch_shape = tf.broadcast_static_shape(
      batch_shape_source, batch_shape_points)

  except ValueError as err:
    raise ValueError((
      "Incompatible batch shapes for `source` and `points`."
      "The batch dimensions for `source` and `points` must be "
      "broadcastable. Received: {}, {}").format(
        source.shape, points.shape)) from err

  source = tf.broadcast_to(source, batch_shape + source_shape)
  points = tf.broadcast_to(points, batch_shape + points_shape)

  return source, points, transform_type, fft_direction, grid_shape


def _validate_enum(value, valid_values, name):
  """Validates that value is in a list of valid values.

  Args:
    value: The value to validate.
    valid_values: The list of valid values.
    name: The name of the argument being validated. This is only used to
      format error messages.

  Returns:
    A valid enum value.

  Raises:
    ValueError: If a value not in the list of valid values was passed.
  """
  if value not in valid_values:
    raise ValueError((
      "The `{}` argument must be one of {}. "
      "Received: {}").format(name, valid_values, value))
  return value


def _complex_dtype(dtype):
  """Returns the corresponding complex dtype for a given real dtype.

  Args:
    dtype: A floating-point dtype, i.e. float32 or float64. Can be a
      string or a `tf.dtypes.DType`.

  Returns:
    The complex dtype corresponding to the given real dtype.
  """
  dtypes = {
    'float32': tf.dtypes.complex64,
    'float64': tf.dtypes.complex128,
    'complex64': tf.dtypes.complex64,
    'complex128': tf.dtypes.complex128
  }
  if isinstance(dtype, tf.dtypes.DType):
    dtype = dtype.name
  return dtypes[dtype]
