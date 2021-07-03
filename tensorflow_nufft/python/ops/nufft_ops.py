# ==============================================================================
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
"""NUFFT ops."""

import numpy as np
import tensorflow as tf

from dlex.utils import check_utils, generic_utils
from dlex.utils import type_utils


cc_lib = tf.load_op_library("dlex/libdlex.so")

nufft = cc_lib.nufft


# def nufft(source,
#           points,
#           transform_type=2,
#           j_sign=-1,
#           epsilon=1e-6,
#           grid_shape=-1):

#     return generic_utils.call_dlex_library('nufft',
#                                            source,
#                                            points,
#                                            transform_type,
#                                            j_sign,
#                                            epsilon,
#                                            grid_shape)


@tf.RegisterGradient("NUFFT")
def _nufft_grad(op, grad):
    """Gradients for `nufft`.

    Args:
        op: The `nufft` `tf.Operation`.
        grad: Gradient with respect to the output of the `nufft` op.

    Returns:
        Gradients with respect to the inputs of `nufft`.
    """
    source = op.inputs[0]
    points = op.inputs[1]

    nufft_rank = points.shape[-1]
    transform_type = op.get_attr('transform_type')
    j_sign = op.get_attr('j_sign')
    epsilon = op.get_attr('epsilon')
    grid_shape = op.get_attr('grid_shape')

    if transform_type == b'type_1':     # nonuniform to uniform
        grad_transform_type = 'type_2'  # uniform to nonuniform
        grad_grid_shape = []
    elif transform_type == b'type_2':
        grad_transform_type = 'type_1'
        grad_grid_shape = source.shape[-nufft_rank:]

    if j_sign == b'positive':
        grad_j_sign = 'negative'
    elif j_sign == b'negative':
        grad_j_sign = 'positive'

    grad_source = nufft(grad,
                        points,
                        transform_type=grad_transform_type,
                        j_sign=grad_j_sign,
                        epsilon=epsilon,
                        grid_shape=grad_grid_shape)

    return [grad_source, None]



def nudft(source, points, transform_type='type_2', j_sign='negative', grid_shape=[]):
    """Compute the non-uniform discrete Fourier transform.

    Args:
        source: A signal-domain tensor with shape (..., *spatial_dims), where
            `...` can be any number of batch dimensions.
        points: The frequency coordinates, in cycles/pixel, where the Fourier
            transform should be evaluated. Must be a tensor of shape
            (..., M, N), where N is the number of spatial dimensions, M is
            the number of coordinates and `...` is any number of batch
            dimensions. The batch dimensions for `source` and `points` must be
            broadcastable.
        transform_type: Type of the transform. Must be 1 (nonuniform to uniform)
            or 2 (uniform to nonuniform).
        j_sign: Sign of the imaginary unit in the exponential. Must be -1
            (signal to frequency domain) or 1 (frequency to signal domain).
        
    Returns:
        The discrete Fourier transform of the input signal-domain tensor,
        evaluated at the specified frequencies.
    """
    def _nudft(inputs):
        src, pts = inputs
        shape = src.shape if transform_type == 'type_2' else grid_shape
        nudft_matrix = _nudft_matrix(
            pts, shape, j_sign=j_sign)
        if transform_type == 'type_1':
            nudft_matrix = tf.transpose(nudft_matrix)
        src_vec = tf.reshape(src, [-1])
        return tf.linalg.matvec(nudft_matrix, src_vec)

    # Validate inputs. This also broadcasts `source` and `points` to equal
    # batch shapes.
    source, points, transform_type, j_sign, grid_shape = _validate_nudft_inputs(
        source, points, transform_type, j_sign, grid_shape)

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


def _nudft_matrix(points, grid_shape, j_sign):
    """Compute the nonuniform Fourier transform matrix.

    Args:
        points: Nonuniform points, in the range [-pi, pi].
        grid_shape: Shape of the gridded tensor.
        j_sign: Sign of the imaginary unit in the exponential. Must be either
            'positive' or 'negative'.

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
        points, r_grid), type_utils.complex_dtype(points.dtype))

    if j_sign == 'positive':
        nudft_matrix = tf.exp(1j * points_grid)
    elif j_sign == 'negative':
        nudft_matrix = tf.exp(-1j * points_grid)

#     TODO: scaling?
#     nudft_matrix = nudft_matrix / (
#         np.sqrt(tf.reduce_prod(grid_shape)) * np.power(np.sqrt(2), rank))
    return nudft_matrix



def _validate_nudft_inputs(source,
                           points,
                           transform_type,
                           j_sign,
                           grid_shape=None,
                           expected_rank=None,
                           expected_grid_shape=None,
                           expected_dtype=None):
    """Validate inputs for non-uniform discrete Fourier transform.

    Args:
        source: An image or signal-domain tensor.
        points: The frequency coordinates.
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
        ValueError: If the batch shapes of `source` and `points` are not broadcastable.
    """
    # Check flags.
    transform_type = check_utils.validate_enum(
        transform_type, {'type_1', 'type_2'}, 'transform_type')
    j_sign = check_utils.validate_enum(
        j_sign, {'positive', 'negative'}, 'j_sign')

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

    return source, points, transform_type, j_sign, grid_shape
