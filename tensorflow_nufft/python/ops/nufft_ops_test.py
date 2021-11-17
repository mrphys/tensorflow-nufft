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
"""Tests for NUFFT ops."""

import functools
import itertools

import numpy as np
import tensorflow as tf

from tensorflow_nufft.python.ops import nufft_ops


def parameterized(**params):
  """Decorates a test to run with multiple parameter combinations.

  All possible combinations (Cartesian product) of the parameters will be
  tested.

  Args:
    params: The keyword parameters to be passed to the subtests. Each keyword
      argument should be a list of values to be tested.

  Returns:
    A decorator which calls the decorated test with multiple parameter
    combinations.
  """
  param_lists = params

  def decorator(func):

    @functools.wraps(func)
    def run(self):

      # Create combinations of the parameters above.
      values = itertools.product(*param_lists.values())
      params = [dict(zip(param_lists.keys(), v)) for v in values]
      # Now call decorated function with each set of parameters.
      for i, p in enumerate(params):
        with self.subTest(**p):
          print(f"[{func.__name__}] - subtest #{i + 1}/{len(params)}: {p}")
          func(self, **p)
    return run
  return decorator


class NUFFTOpsTest(tf.test.TestCase):
  """Test case for NUFFT functions."""

  def test_parallel_iteration(self):
    with tf.device('/cpu:0'):
      rank = 2
      num_points = 20000
      grid_shape = [128] * rank
      batch_size = 100
      rng = tf.random.Generator.from_seed(10)
      points = rng.uniform([batch_size, num_points, rank], minval=-np.pi, maxval=np.pi)
      source = tf.complex(tf.ones([batch_size, num_points]),
                          tf.zeros([batch_size, num_points]))
      @tf.function
      def parallel_nufft_adjoint(source, points):
        def nufft_adjoint(inputs):
          src, pts = inputs
          return nufft_ops.nufft(src, pts, grid_shape=grid_shape,
                                transform_type='type_1',
                                fft_direction='backward')
        return tf.map_fn(nufft_adjoint, [source, points],
                        parallel_iterations=4,
                        fn_output_signature=tf.TensorSpec(grid_shape, tf.complex64))
      
      result = parallel_nufft_adjoint(source, points)
      print(result.shape)

  # @parameterized(grid_shape=[[10, 16], [10, 10, 8]],
  #                source_batch_shape=[[], [2, 4]],
  #                points_batch_shape=[[], [2, 1], [1, 4]],
  #                transform_type=['type_1', 'type_2'],
  #                fft_direction=['forward', 'backward'],
  #                dtype=[tf.dtypes.complex64, tf.dtypes.complex128],
  #                device=['/cpu:0']) # TODO: re-enable GPU checks.
  # def test_nufft(self,  # pylint: disable=missing-param-doc
  #                grid_shape=None,
  #                source_batch_shape=None,
  #                points_batch_shape=None,
  #                transform_type=None,
  #                fft_direction=None,
  #                dtype=None,
  #                device=None):
  #   """Test NUFFT op result and gradients against naive NUDFT results."""
  #   # pylint: disable=unexpected-keyword-arg

  #   # Set random seed.
  #   tf.random.set_seed(0)

  #   # Skip float64 GPU tests because it has limited support on testing hardware.
  #   if dtype == tf.dtypes.complex128 and device == '/gpu:0':
  #     return

  #   with tf.device(device):

  #     rank = len(grid_shape)
  #     num_points = tf.math.reduce_prod(grid_shape)

  #     # Generate random signal and points.
  #     if transform_type == 'type_1':    # nonuniform to uniform
  #       source_shape = source_batch_shape + [num_points]
  #     elif transform_type == 'type_2':  # uniform to nonuniform
  #       source_shape = source_batch_shape + grid_shape

  #     source = tf.Variable(tf.dtypes.complex(
  #       tf.random.uniform(
  #         source_shape, minval=-0.5, maxval=0.5, dtype=dtype.real_dtype),
  #       tf.random.uniform(
  #         source_shape, minval=-0.5, maxval=0.5, dtype=dtype.real_dtype)))

  #     points_shape = points_batch_shape + [num_points, rank]
  #     points = tf.Variable(tf.random.uniform(
  #       points_shape, minval=-np.pi, maxval=np.pi,
  #       dtype=dtype.real_dtype))

  #     with tf.GradientTape(persistent=True) as tape:

  #       result_nufft = nufft_ops.nufft(
  #         source, points,
  #         grid_shape=grid_shape,
  #         transform_type=transform_type,
  #         fft_direction=fft_direction)

  #       result_nudft = nufft_ops.nudft(
  #         source, points,
  #         grid_shape=grid_shape,
  #         transform_type=transform_type,
  #         fft_direction=fft_direction)

  #     # Compute gradients.
  #     grad_nufft = tape.gradient(result_nufft, source)
  #     grad_nudft = tape.gradient(result_nudft, source)

  #     tol = 1.e-3
  #     self.assertAllClose(result_nudft, result_nufft,
  #                         rtol=tol, atol=tol)
  #     self.assertAllClose(grad_nufft, grad_nudft,
  #                         rtol=tol, atol=tol)


  # @parameterized(grid_shape=[[128, 128], [128, 128, 128]],
  #                dtype=[tf.complex64, tf.complex128],
  #                device=['/cpu:0', '/gpu:0'])
  # def test_interp(self, grid_shape, dtype, device): # pylint: disable=missing-function-docstring

  #   if dtype == tf.complex128 and 'gpu' in device:
  #     return

  #   tf.random.set_seed(0)

  #   batch_shape = []
  #   num_points = 100
  #   rank = len(grid_shape)

  #   source = tf.complex(tf.ones(batch_shape + grid_shape, dtype.real_dtype),
  #                       tf.zeros(batch_shape + grid_shape, dtype.real_dtype))
  #   points = tf.random.uniform(batch_shape + [num_points, rank], # pylint: disable=unexpected-keyword-arg
  #     minval=-np.pi, maxval=np.pi, dtype=dtype.real_dtype)

  #   with tf.device(device):
  #     target = nufft_ops.interp(source, points, tol=1e-4)

  #   expected = tf.complex(
  #     tf.ones(batch_shape + [num_points], dtype.real_dtype),
  #     tf.zeros(batch_shape + [num_points], dtype.real_dtype))

  #   self.assertAllEqual(target.shape, expected.shape)
  #   self.assertAllClose(target, expected, rtol=1e-4, atol=1e-4)


  # @parameterized(grid_shape=[[64, 64], [64, 64, 64]],
  #                dtype=[tf.complex64, tf.complex128],
  #                device=['/cpu:0', '/gpu:0'])
  # def test_spread(self, grid_shape, dtype, device): # pylint: disable=missing-function-docstring

  #   if dtype == tf.complex128 and 'gpu' in device:
  #     return

  #   tf.random.set_seed(0)

  #   batch_shape = []
  #   num_points = 1
  #   for s in grid_shape:
  #     num_points *= s
  #   rank = len(grid_shape)

  #   source = tf.complex(tf.ones(batch_shape + [num_points], dtype.real_dtype),
  #                       tf.zeros(batch_shape + [num_points], dtype.real_dtype))
  #   points = tf.random.uniform(batch_shape + [num_points, rank], # pylint: disable=unexpected-keyword-arg
  #     minval=-np.pi, maxval=np.pi, dtype=dtype.real_dtype)

  #   with tf.device(device):
  #     target = nufft_ops.spread(source, points, grid_shape)

  #   # We are spreading from random coordinates, so there will be significant
  #   # deviations from 1. However, check that values are within a ballpark and
  #   # that the average is 1.0.
  #   self.assertAllInRange(tf.math.real(target), 0.0, 3.0)
  #   self.assertAllClose(tf.math.reduce_mean(target), 1.0)


  # @parameterized(device=['/cpu:0', '/gpu:0'])
  # def test_interp_batch(self, device): # pylint: disable=missing-function-docstring

  #   tf.random.set_seed(0)

  #   dtype = tf.complex64
  #   grid_shape = [64, 96]
  #   batch_size = 4
  #   num_points = 100
  #   rank = len(grid_shape)

  #   sources = []
  #   for i in range(batch_size):
  #     sources.append(tf.complex(i * tf.ones(grid_shape, dtype.real_dtype),
  #                               tf.zeros(grid_shape, dtype.real_dtype)))
  #   source = tf.stack(sources)

  #   points = tf.random.uniform( # pylint: disable=unexpected-keyword-arg
  #     [batch_size, num_points, rank],
  #     minval=-np.pi, maxval=np.pi, dtype=dtype.real_dtype) # pylint: disable=unexpected-keyword-arg

  #   with tf.device(device):
  #     target = nufft_ops.interp(source, points, tol=1e-4)

  #   self.assertAllEqual(target.shape, [batch_size, num_points])
  #   for i in range(batch_size):
  #     expected = tf.complex(
  #       i * tf.ones([num_points], dtype.real_dtype),
  #       tf.zeros([num_points], dtype.real_dtype))
  #     self.assertAllClose(target[i, ...], expected, rtol=1e-4, atol=1e-4)


  # @parameterized(device=['/cpu:0', '/gpu:0'])
  # def test_spread_batch(self, device): # pylint: disable=missing-function-docstring

  #   tf.random.set_seed(0)

  #   dtype = tf.complex64
  #   grid_shape = [64, 96]
  #   batch_size = 4
  #   num_points = 1
  #   for s in grid_shape:
  #     num_points *= s
  #   rank = len(grid_shape)

  #   sources = []
  #   for i in range(batch_size):
  #     sources.append(tf.complex(i * tf.ones([num_points], dtype.real_dtype),
  #                               tf.zeros([num_points], dtype.real_dtype)))
  #   source = tf.stack(sources)

  #   points = tf.random.uniform( # pylint: disable=unexpected-keyword-arg
  #     [batch_size, num_points, rank],
  #     minval=-np.pi, maxval=np.pi, dtype=dtype.real_dtype)

  #   with tf.device(device):
  #     target = nufft_ops.spread(source, points, grid_shape, tol=1e-4)

  #   self.assertAllEqual(target.shape, [batch_size] + grid_shape)
  #   for i in range(batch_size):
  #     self.assertAllClose(tf.math.reduce_mean(target[i, ...]), i,
  #                         rtol=1e-4, atol=1e-4)


  # @parameterized(transform_type=['type_1', 'type_2'],
  #                which=['source', 'points'],
  #                device=['/cpu:0', '/gpu:0'])
  # def test_nufft_different_batch_ranks(self, transform_type, which, device): # pylint: disable=missing-function-docstring
  #   # pylint: disable=unexpected-keyword-arg

  #   # Set random seed.
  #   tf.random.set_seed(0)

  #   grid_shape = [24, 24]
  #   if which == 'points':
  #     source_batch_shape = [2, 4]
  #     points_batch_shape = [1]
  #   else:
  #     points_batch_shape = [2, 4]
  #     source_batch_shape = [1]
  #   dtype = tf.complex64
  #   fft_direction='forward'

  #   rank = len(grid_shape)
  #   num_points = tf.math.reduce_prod(grid_shape)

  #   with tf.device(device):
  #     # Generate random signal and points.
  #     if transform_type == 'type_1':    # nonuniform to uniform
  #       source_shape = source_batch_shape + [num_points]
  #     elif transform_type == 'type_2':  # uniform to nonuniform
  #       source_shape = source_batch_shape + grid_shape

  #     source = tf.Variable(tf.dtypes.complex(
  #       tf.random.uniform(
  #         source_shape, minval=-0.5, maxval=0.5, dtype=dtype.real_dtype),
  #       tf.random.uniform(
  #         source_shape, minval=-0.5, maxval=0.5, dtype=dtype.real_dtype)))

  #     points_shape = points_batch_shape + [num_points, rank]
  #     points = tf.Variable(tf.random.uniform(
  #       points_shape, minval=-np.pi, maxval=np.pi,
  #       dtype=dtype.real_dtype))

  #     with tf.GradientTape(persistent=True) as tape:

  #       result_nufft = nufft_ops.nufft(
  #         source, points,
  #         grid_shape=grid_shape,
  #         transform_type=transform_type,
  #         fft_direction=fft_direction)

  #       result_nudft = nufft_ops.nudft(
  #         source, points,
  #         grid_shape=grid_shape,
  #         transform_type=transform_type,
  #         fft_direction=fft_direction)

  #     # Compute gradients.
  #     grad_nufft = tape.gradient(result_nufft, source)
  #     grad_nudft = tape.gradient(result_nudft, source)

  #     tol = 1.e-3
  #     self.assertAllClose(result_nudft, result_nufft,
  #                         rtol=tol, atol=tol)
  #     self.assertAllClose(grad_nufft, grad_nudft,
  #                         rtol=tol, atol=tol)


  # @parameterized(device=['/cpu:0', '/gpu:0'])
  # def test_interp_3d_many_points(self, device): # pylint: disable=missing-param-doc
  #   """Test 3D interpolation with a large points array."""
  #   for _ in range(5):
  #     # We repeat this test several times because non-deterministic behaviour
  #     # has been observed with this kind of data, so make sure it's not
  #     # happening.
  #     with tf.device(device):
  #       num_points = 3000000
  #       rng = tf.random.Generator.from_seed(0)
  #       points = rng.uniform([num_points, 3], minval=-np.pi, maxval=np.pi)
  #       source = tf.complex(tf.ones([128, 128, 128]),
  #                           tf.zeros([128, 128, 128]))
  #       result = nufft_ops.interp(source, points)
  #       self.assertAllClose(tf.math.real(result), tf.ones([num_points]),
  #                           rtol=1e-5, atol=1e-5)


  # def test_static_shape(self): # pylint: disable=missing-function-docstring

  #   tf.compat.v1.disable_v2_behavior()
  #   self._assert_static_shapes([100, 100], [1000, 2], [1000])
  #   self._assert_static_shapes([100, 100], [None, 2], [None])
  #   self._assert_static_shapes([None, None], [None, 2], [None])
  #   self._assert_static_shapes([None, None], [None, None], None)
  #   self._assert_static_shapes([8, 100, 100], [1000, 2], [8, 1000])
  #   self._assert_static_shapes([8, 100, 100], [None, 1000, 2], [8, 1000])
  #   self._assert_static_shapes([None, 100, 100], [None, 1000, 2], [None, 1000])
  #   self._assert_static_shapes([60, 100, 100], [1000, 3], [1000])

  #   self._assert_static_shapes([None, 40, 1000], [None, 1000, 2],
  #                              [None, 40, 200, 200], 'type_1', [200, 200])
  #   self._assert_static_shapes([None, None], [None, 1000, 2],
  #                              [None, 100, 200], 'type_1', [100, 200])
  #   self._assert_static_shapes([None, None], [None, 1000, 2],
  #                              [None, None, None], 'type_1', 'tensor')

  #   self._assert_static_raises([8, 100, 100], [6, 1000, 2],
  #                              regex="Dimensions must be equal")
  #   self._assert_static_raises([50, 50, 50, 50], [500, 4],
  #                              regex="Dimension must be 1, 2 or 3")


  # def _assert_static_shapes(self, source_shape, points_shape, target_shape, # pylint: disable=missing-function-docstring
  #                           transform_type='type_2', grid_shape=None):

  #   source = tf.compat.v1.placeholder(tf.complex64, source_shape)
  #   points = tf.compat.v1.placeholder(tf.float32, points_shape)
  #   if grid_shape == 'tensor':
  #     grid_shape = tf.compat.v1.placeholder(tf.int32, [2])
  #   target = nufft_ops.nufft(source, points,
  #                            grid_shape=grid_shape,
  #                            transform_type=transform_type)
  #   if target.shape.rank is not None:
  #     self.assertEqual(target.shape.as_list(), target_shape)
  #   else:
  #     self.assertEqual(target_shape, None)


  # def _assert_static_raises(self, source_shape, points_shape, # pylint: disable=missing-function-docstring
  #                           transform_type='type_2', grid_shape=None,
  #                           regex=None):

  #   source = tf.compat.v1.placeholder(tf.complex64, source_shape)
  #   points = tf.compat.v1.placeholder(tf.float32, points_shape)
  #   if grid_shape == 'tensor':
  #     grid_shape = tf.compat.v1.placeholder(tf.int32, [2])
  #   if regex is not None:
  #     with self.assertRaisesRegex(ValueError, regex):
  #       nufft_ops.nufft(source, points,
  #                       transform_type=transform_type,
  #                       grid_shape=grid_shape)
  #   else:
  #     with self.assertRaises(ValueError):
  #       nufft_ops.nufft(source, points,
  #                       transform_type=transform_type,
  #                       grid_shape=grid_shape)


class NUFFTOpsBenchmark(tf.test.Benchmark):
  """Benchmark for NUFFT functions."""

  # source_shape, points_shape, transform_type, grid_shape
  cases = [
    ([256, 256], [200000, 2], 'type_2', None),
    ([16, 256, 256], [200000, 2], 'type_2', None),
    ([16, 256, 256], [16, 200000, 2], 'type_2', None),
    ([200000], [200000, 2], 'type_1', [256, 256]),
    ([16, 200000], [200000, 2], 'type_1', [256, 256]),
    ([16, 200000], [16, 200000, 2], 'type_1', [256, 256]),
    ([128, 128, 128], [800000, 3], 'type_2', None),
    ([800000], [800000, 3], 'type_1', [128, 128, 128])
  ]

  def benchmark_nufft(self):
    """Benchmark NUFFT op."""

    source_shape = [256, 256]
    points_shape = [65536, 2]

    dtype = tf.dtypes.complex64

    rng = np.random.default_rng(0)

    def random_array(shape):
      return rng.random(shape, dtype=dtype.real_dtype.name) - 0.5

    devices = ['/cpu:0']
    if tf.test.gpu_device_name():
      devices.append(tf.test.gpu_device_name())

    results = []
    headers = []

    for d, device in enumerate(devices):

      results.append([])
      headers.append([])

      for source_shape, points_shape, transform_type, grid_shape in self.cases:

        with tf.Graph().as_default(), \
            tf.compat.v1.Session(config=tf.test.benchmark_config()) as sess, \
            tf.device(device):

          source = tf.Variable(
            random_array(source_shape) + random_array(source_shape) * 1j)
          points = tf.Variable(
            random_array(points_shape) * 2.0 * np.pi)

          self.evaluate(tf.compat.v1.global_variables_initializer())

          target = nufft_ops.nufft(source,
                                   points,
                                   grid_shape=grid_shape,
                                   transform_type=transform_type)

          result = self.run_op_benchmark(
            sess,
            target,
            burn_iters=2,
            min_iters=50,
            store_memory_usage=True,
            extras={
              'source_shape': source_shape,
              'points_shape': points_shape,
              'transform_type': transform_type,
              'grid_shape': grid_shape
            })

        result.update(result['extras'])
        result.pop('extras')
        headers[d] = list(result.keys())
        results[d].append(list(result.values()))

    for r, h in zip(results, headers):
      try:
        from tabulate import tabulate # pylint: disable=import-outside-toplevel
        print(tabulate(r, headers=h))
      except ModuleNotFoundError:
        pass


if __name__ == '__main__':
  tf.test.main()
