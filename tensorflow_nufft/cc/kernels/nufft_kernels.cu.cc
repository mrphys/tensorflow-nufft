/* Copyright 2021 The TensorFlow NUFFT Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"

#include "tensorflow_nufft/cc/kernels/nufft_kernels.h"
#include "tensorflow_nufft/cc/kernels/nufft_options.h"
#include "tensorflow_nufft/cc/kernels/nufft_plan.h"


namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace nufft {

template<typename FloatType>
struct DoNUFFT<GPUDevice, FloatType> : DoNUFFTBase<GPUDevice, FloatType> {
  Status operator()(OpKernelContext* ctx,
                    TransformType type,
                    int rank,
                    FftDirection fft_direction,
                    int num_transforms,
                    FloatType tol,
                    OpType op_type,
                    int64_t batch_rank,
                    int64_t* source_batch_dims,
                    int64_t* points_batch_dims,
                    int64_t* num_modes,
                    int64_t num_points,
                    FloatType* points,
                    Complex<GPUDevice, FloatType>* source,
                    Complex<GPUDevice, FloatType>* target) {
    return this->compute(
        ctx, type, rank, fft_direction, num_transforms, tol, op_type,
        batch_rank, source_batch_dims, points_batch_dims,
        num_modes, num_points, points, source, target);
  }
};

// Explicitly instantiate functors for the types of OpKernels registered.
template struct DoNUFFT<GPUDevice, float>;
template struct DoNUFFT<GPUDevice, double>;

}  // namespace nufft
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
