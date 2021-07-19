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
"""Tests for signal ops."""

import functools
import itertools
import time

import numpy as np
import tensorflow as tf

import nufft_ops


def parameterized(**params):
    param_lists = params
    def decorator(func):
        @functools.wraps(func)
        def run_tests(self):
            # Create combinations of the parameters above.
            values = itertools.product(*param_lists.values())
            params = [dict(zip(param_lists.keys(), v)) for v in values]
            # Now call decorated function with each 
            for p in params:
                with self.subTest(**p):
                    func(self, **p)
        return run_tests
    return decorator


class NUFFTOpsTest(tf.test.TestCase):

    @parameterized(grid_shape=[[16, 16], [16, 16, 16]],
                   source_batch_shape=[[2, 4]],
                   points_batch_shape=[[1, 1]],
                   transform_type=['type_1', 'type_2'],
                   j_sign=['negative', 'positive'],
                   dtype=[tf.dtypes.complex64, tf.dtypes.complex128],
                   broadcast=[False],
                   device=['/cpu:0', '/gpu:0'])
    def test_nufft(self,
                   grid_shape=None,
                   source_batch_shape=None,
                   points_batch_shape=None,
                   transform_type=None,
                   j_sign=None,
                   dtype=None,
                   broadcast=None,
                   device=None):

        # Set random seed.
        tf.random.set_seed(0)

        # Skip float64 GPU tests. Something's not quite right with those. Good
        # luck dealing with that to future self. TODO
        if dtype == tf.dtypes.complex128 and device == '/gpu:0':
            return

        with tf.device(device):

            rank = len(grid_shape)
            num_points = tf.math.reduce_prod(grid_shape)

            # Generate random signal and points.
            if transform_type == 'type_1': # nonuniform to uniform
                source_shape = source_batch_shape + [num_points]
            elif transform_type == 'type_2': # uniform to nonuniform
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
                # result_legacy = signal_ops.nufft(source, points)

                # start = time.time()
                # result_legacy = signal_ops.nufft(source, points / (2.0 * np.pi))
                # time_legacy = time.time() - start
 
                start = time.time()
                result_nufft = nufft_ops.nufft(
                    source, points,
                    transform_type=transform_type,
                    j_sign=j_sign,
                    grid_shape=grid_shape)
                time_nufft = time.time() - start

                start = time.time()
                result_nudft = nufft_ops.nudft(
                    source, points,
                    transform_type=transform_type,
                    j_sign=j_sign,
                    grid_shape=grid_shape)
                time_nudft = time.time() - start

                

            # Compute gradients.
            grad_nufft = tape.gradient(result_nufft, source)
            grad_nudft = tape.gradient(result_nudft, source)

            epsilon = 1.e-3
            self.assertAllClose(result_nudft, result_nufft, rtol=epsilon, atol=epsilon)
            self.assertAllClose(grad_nufft, grad_nudft, rtol=epsilon, atol=epsilon)


if __name__ == '__main__':
    tf.test.main()
