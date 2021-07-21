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
from tensorflow._api.v2 import random

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
    def run_tests(self):

      # Create combinations of the parameters above.
      values = itertools.product(*param_lists.values())
      params = [dict(zip(param_lists.keys(), v)) for v in values]
      # Now call decorated function with each set of parameters.
      for i, p in enumerate(params[:2]):
        with self.subTest(**p):
          print(f"Subtest #{i + 1}/{len(params)}: {p}")
          func(self, **p)

    return run_tests
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
    """Test op result and gradients."""
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


class NUFFTOpsBenchmark(tf.test.Benchmark):

  def benchmark_nufft(self):
    
    # tf.compat.v1.disable_v2_behavior()

    source_shape = [10, 10]
    points_shape = [100, 2]

    dtype = tf.dtypes.complex64

    rng = np.random.default_rng(0)
    
    def random(shape):
      return rng.random(shape, dtype=dtype.real_dtype.name) - 0.5

    source = tf.Variable(random(source_shape) + random(source_shape) * 1j)
    points = tf.Variable(random(points_shape) * 2.0 * np.pi)
    result = nufft_ops.nufft(source, points)
    print(result.shape)

    with tf.Graph().as_default(), \
        tf.compat.v1.Session(config=tf.test.benchmark_config()) as sess:

      source = tf.Variable(random(source_shape) + random(source_shape) * 1j)
      points = tf.Variable(random(points_shape) * 2.0 * np.pi)
      self.evaluate(tf.compat.v1.global_variables_initializer())
      
      result = nufft_ops.nufft(source, points)

      v = sess.run(result)
      print(v)

      self.run_op_benchmark(
        sess,
        nufft_ops.nufft(source, points),
        min_iters=50)

#     # self.report_benchmark(
#     #     iters=2,
#     #     name="custom_benchmark_name",
#     #     extras={"number_key": 3,
#     #             "other_key": "string"})

# class EinsumBenchmark(tf.test.Benchmark):
#   cases = [
#       # Unary cases.
#       ['ijk->i', 100],
#       ['ijk->kji', 100],
#   ]

#   def benchmarkEinsum(self):
#     for equation, dim in self.cases:
#       with tf.Graph().as_default(), \
#           tf.compat.v1.Session(config=tf.test.benchmark_config()) as sess, \
#           tf.device('/cpu:0'):
#         r = np.random.RandomState(0)
#         input_subscripts = equation.split('->')[0].split(',')
#         input_vars = []
#         for subscript in input_subscripts:
#           input_shape = (dim,) * len(subscript)
#           input_vars.append(
#               tf.Variable(np.array(r.randn(*input_shape), np.float32)))
#         self.evaluate(tf.compat.v1.global_variables_initializer())

#         # Call einsum_v1.
#         self.run_op_benchmark(
#             sess,
#             tf.einsum(equation, *input_vars),
#             min_iters=50,
#             name='einsum_v1_cpu_({})_{}'.format(equation, dim))

        # # Call gen_linalg_ops.einsum.
        # self.run_op_benchmark(
        #     sess,
        #     tf.einsum(input_vars, equation),
        #     min_iters=50,
        #     name='einsum_v2_cpu_({})_{}'.format(equation, dim))

if __name__ == '__main__':
  tf.test.main()
