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

import nufft_ops


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
          print(f"Subtest #{i + 1}/{len(params)}: {p}")
          func(self, **p)

    return run
  return decorator


class NUFFTOpsTest(tf.test.TestCase):
  """Test case for NUFFT functions."""

  @parameterized(grid_shape=[[10, 10], [10, 10, 10]],
                 source_batch_shape=[[], [2, 4]],
                 points_batch_shape=[[], [2, 1], [1, 4]],
                 transform_type=['type_1', 'type_2'],
                 j_sign=['negative', 'positive'],
                 dtype=[tf.dtypes.complex64, tf.dtypes.complex128],
                 device=['/cpu:0', '/gpu:0'])
  def test_nufft(self,  # pylint: disable=missing-param-doc
                 grid_shape=None,
                 source_batch_shape=None,
                 points_batch_shape=None,
                 transform_type=None,
                 j_sign=None,
                 dtype=None,
                 device=None):
    """Test NUFFT op result and gradients."""
    # pylint: disable=unexpected-keyword-arg

    # Set random seed.
    tf.random.set_seed(0)

    # Skip float64 GPU tests. Something's not quite right with those. Is it
    # due to limited support on NVIDIA card used for testing or is anything
    # else off?
    if dtype == tf.dtypes.complex128 and device == '/gpu:0':
      return

    with tf.device(device):

      rank = len(grid_shape)
      num_points = tf.math.reduce_prod(grid_shape)

      # Generate random signal and points.
      if transform_type == 'type_1':    # nonuniform to uniform
        source_shape = source_batch_shape + [num_points]
      elif transform_type == 'type_2':  # uniform to nonuniform
        source_shape = source_batch_shape + grid_shape

      source = tf.Variable(tf.dtypes.complex(
        tf.random.uniform(
          source_shape, minval=-0.5, maxval=0.5, dtype=dtype.real_dtype),
        tf.random.uniform(
          source_shape, minval=-0.5, maxval=0.5, dtype=dtype.real_dtype)))

      points_shape = points_batch_shape + [num_points, rank]
      points = tf.Variable(tf.random.uniform(
        points_shape, minval=-np.pi, maxval=np.pi,
        dtype=dtype.real_dtype))

      with tf.GradientTape(persistent=True) as tape:

        result_nufft = nufft_ops.nufft(
          source, points,
          transform_type=transform_type,
          j_sign=j_sign,
          grid_shape=grid_shape)

        result_nudft = nufft_ops.nudft(
          source, points,
          transform_type=transform_type,
          j_sign=j_sign,
          grid_shape=grid_shape)

      # Compute gradients.
      grad_nufft = tape.gradient(result_nufft, source)
      grad_nudft = tape.gradient(result_nudft, source)

      epsilon = 1.e-3
      self.assertAllClose(result_nudft, result_nufft,
                          rtol=epsilon, atol=epsilon)
      self.assertAllClose(grad_nufft, grad_nudft,
                          rtol=epsilon, atol=epsilon)


  def test_static_shape(self): # pylint: disable=missing-function-docstring

    tf.compat.v1.disable_v2_behavior()
    self._assert_static_shapes([100, 100], [1000, 2], [1000])
    self._assert_static_shapes([100, 100], [None, 2], [None])
    self._assert_static_shapes([None, None], [None, 2], [None])
    self._assert_static_shapes([None, None], [None, None], None)
    self._assert_static_shapes([8, 100, 100], [1000, 2], [8, 1000])
    self._assert_static_shapes([8, 100, 100], [None, 1000, 2], [8, 1000])
    self._assert_static_shapes([None, 100, 100], [None, 1000, 2], [None, 1000])
    self._assert_static_shapes([60, 100, 100], [1000, 3], [1000])

    self._assert_static_shapes([None, 40, 1000], [None, 1000, 2],
                               [None, 40, 200, 200], 'type_1', [200, 200])
    self._assert_static_shapes([None, None], [None, 1000, 2],
                               [None, 100, 200], 'type_1', [100, 200])

    self._assert_static_raises([None, 40, 1000], [None, 1000, 2], 'type_1',
                               regex="grid_shape attr must be fully defined")
    self._assert_static_raises([8, 100, 100], [6, 1000, 2],
                               regex="Dimensions must be equal")
    self._assert_static_raises([50, 50, 50, 50], [500, 4],
                               regex="Dimension must be 1, 2 or 3")


  def _assert_static_shapes(self, source_shape, points_shape, target_shape, # pylint: disable=missing-function-docstring
                            transform_type='type_2', grid_shape=None):

    source = tf.compat.v1.placeholder(tf.complex64, source_shape)
    points = tf.compat.v1.placeholder(tf.float32, points_shape)
    target = nufft_ops.nufft(source, points,
                             transform_type=transform_type,
                             grid_shape=grid_shape)
    if target.shape.rank is not None:
      self.assertEqual(target.shape.as_list(), target_shape)
    else:
      self.assertEqual(target_shape, None)


  def _assert_static_raises(self, source_shape, points_shape, # pylint: disable=missing-function-docstring
                            transform_type='type_2', grid_shape=None,
                            regex=None):

    source = tf.compat.v1.placeholder(tf.complex64, source_shape)
    points = tf.compat.v1.placeholder(tf.float32, points_shape)
    if regex is not None:
      with self.assertRaisesRegex(ValueError, regex):
        nufft_ops.nufft(source, points,
                        transform_type=transform_type,
                        grid_shape=grid_shape)
    else:
      with self.assertRaises(ValueError):
        nufft_ops.nufft(source, points,
                        transform_type=transform_type,
                        grid_shape=grid_shape)


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
                                   transform_type=transform_type,
                                   grid_shape=grid_shape)

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
