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

import itertools
import time

import numpy as np
import tensorflow as tf

from dlex.ops import signal_ops
from dlex.ops import spectral_ops


class SpectralOpsTest(tf.test.TestCase): # pylint: disable=missing-class-docstring

    def test_nufft(self):
        """Test NUFFT."""

        tf.random.set_seed(0)

        params = {
            'grid_shape': [[16, 16], [16, 16, 16]], # 2D, 3D
            'batch_shape': [[]], # With/without batch dimensions.
            'transform_type': ['type_1', 'type_2'],
            'j_sign': ['negative', 'positive'],
            'dtype': [tf.dtypes.complex64, tf.dtypes.complex128],
            'broadcast': [False],
        }

        # Create combinations of the parameters above.
        values = itertools.product(*params.values())
        params = [dict(zip(params.keys(), v)) for v in values]

        for p in params:
            with self.subTest(**p):

                grid_shape = p['grid_shape']
                batch_shape = p['batch_shape']
                transform_type = p['transform_type']
                j_sign=p['j_sign']
                dtype = p['dtype']
                rank = len(grid_shape)
                num_points = tf.math.reduce_prod(grid_shape)

                # Generate random signal and points.
                if transform_type == 'type_1': # nonuniform to uniform
                    source_shape = batch_shape + [num_points]
                elif transform_type == 'type_2': # uniform to nonuniform
                    source_shape = batch_shape + grid_shape

                source = tf.Variable(tf.dtypes.complex(
                    tf.random.uniform(
                        source_shape, minval=-0.5, maxval=0.5, dtype=dtype.real_dtype),
                    tf.random.uniform(
                        source_shape, minval=-0.5, maxval=0.5, dtype=dtype.real_dtype)))

                points_shape = batch_shape + [num_points, rank]
                points = tf.Variable(tf.random.uniform(
                    points_shape, minval=-np.pi, maxval=np.pi,
                    dtype=dtype.real_dtype))

                with tf.GradientTape(persistent=True) as tape:
                    # result_legacy = signal_ops.nufft(source, points)

                    # start = time.time()
                    # result_legacy = signal_ops.nufft(source, points / (2.0 * np.pi))
                    # time_legacy = time.time() - start
                    start = time.time()
                    result_nudft = spectral_ops.nudft(
                        source, points,
                        transform_type=transform_type,
                        j_sign=j_sign,
                        grid_shape=grid_shape)
                    time_nudft = time.time() - start

                    if transform_type == 'type_1':
                        source_t = source
                    if transform_type == 'type_2':
                        source_t = tf.transpose(source) # TODO: build into op

                    start = time.time()
                    result_nufft_t = spectral_ops.nufft(
                        source_t, points,
                        transform_type=transform_type,
                        j_sign=j_sign,
                        grid_shape=grid_shape)
                    time_nufft = time.time() - start

                    if transform_type == 'type_1':
                        result_nufft = tf.transpose(result_nufft_t) # TODO: build into op
                    elif transform_type == 'type_2':
                        result_nufft = result_nufft_t

                # Compute gradients.
                grad_nufft = tape.gradient(result_nufft, source)
                grad_nudft = tape.gradient(result_nudft, source)

                epsilon = 1.e-3
                self.assertAllClose(result_nudft, result_nufft, rtol=epsilon, atol=epsilon)
                self.assertAllClose(grad_nufft, grad_nudft, rtol=epsilon, atol=epsilon)

                # help(spectral_ops.nufft)

        # print(grad_nudft)
        # print(grad_nufft)
        # print(result_legacy)
        # print(result_nufft)

        # import finufft
        
        # import pycuda.autoinit
        # from pycuda.gpuarray import GPUArray, to_gpu
        # from cufinufft import cufinufft

        # # for ii in np.arange(0, num_points):
		# #     result_ref[ii] = np.sum(fk * np.exp(1j*(Ks*xj[ii]+Kt*yj[ii])))
        # # result_ref = np.
        # start = time.time()
        # result_fi = finufft.nufft3d2(
        #     points[..., 0].numpy() * 2.0 * np.pi,
        #     points[..., 1].numpy() * 2.0 * np.pi,
        #     points[..., 2].numpy() * 2.0 * np.pi,
        #     source.numpy()) / (1.0541 * 3 * 4 * 20)
        # time_fi = time.time() - start

        # x_gpu = to_gpu(points[..., 0].numpy() * 2.0 * np.pi)
        # y_gpu = to_gpu(points[..., 1].numpy() * 2.0 * np.pi)
        # z_gpu = to_gpu(points[..., 2].numpy() * 2.0 * np.pi)
        # fk_gpu = to_gpu(source.numpy())

        # # Allocate memory for the nonuniform coefficients on the GPU.
        # c_gpu = GPUArray((num_points.numpy(),), dtype=np.complex64)

        # start = time.time()

        # # Initialize the plan and set the points.
        # plan = cufinufft(2, (20, 20, 20), 1, eps=1.0e-6, dtype=np.float32)
        # plan.set_pts(
        #     x_gpu,
        #     y_gpu,
        #     z_gpu)

        # # Execute the plan, reading from the uniform grid fk c and storing the result
        # # in c_gpu.
        # plan.execute(c_gpu, fk_gpu)

        # # Retreive the result from the GPU.
        # time_cufi = time.time() - start

        # result_cufi = c_gpu.get() / (1.0541 * 3 * 4 * 20)

        # print(time_legacy, time_nudft, time_fi, time_cufi)
        # self.assertAllClose(result_nudft, result_fi, atol=1e-5, rtol=1e-5)
        # self.assertAllClose(result_nudft, result_cufi, atol=1e-5, rtol=1e-5)
