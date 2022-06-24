/* Copyright 2021 University College London. All Rights Reserved.

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

#ifndef TENSORFLOW_NUFFT_CC_KERNELS_NUFFT_KERNELS_H_
#define TENSORFLOW_NUFFT_CC_KERNELS_NUFFT_KERNELS_H_

#include <complex>
#include <cstdint>
#include <memory>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/status.h"

#include "tensorflow_nufft/cc/kernels/nufft_options.h"
#include "tensorflow_nufft/cc/kernels/nufft_plan.h"


namespace tensorflow {
namespace nufft {

enum class OpType { NUFFT, INTERP, SPREAD };

template<typename Device, typename FloatType>
using Complex = typename ComplexType<Device, FloatType>::Type;

template<typename Device, typename FloatType>
struct DoNUFFTBase {
  Status compute(OpKernelContext* ctx,
                 TransformType type,
                 int rank,
                 FftDirection fft_direction,
                 int num_transforms,
                 FloatType tol,
                 int max_batch_size,
                 OpType op_type,
                 int64_t batch_rank,
                 int64_t* source_batch_dims,
                 int64_t* points_batch_dims,
                 int64_t* num_modes,
                 int64_t num_points,
                 FloatType* points,
                 Complex<Device, FloatType>* source,
                 Complex<Device, FloatType>* target) {
    // Number of coefficients.
    int num_coeffs = 1;
    for (int d = 0; d < rank; d++) {
      num_coeffs *= num_modes[d];
    }

    // Number of calls to FINUFFT execute.
    int num_calls = 1;
    for (int d = 0; d < batch_rank; d++) {
      num_calls *= points_batch_dims[d];
    }

    // Factors to transform linear indices to subindices and viceversa.
    gtl::InlinedVector<int, 8> source_batch_factors(batch_rank);
    for (int d = 0; d < batch_rank; d++) {
      source_batch_factors[d] = 1;
      for (int j = d + 1; j < batch_rank; j++) {
        source_batch_factors[d] *= source_batch_dims[j];
      }
    }

    gtl::InlinedVector<int, 8> points_batch_factors(batch_rank);
    for (int d = 0; d < batch_rank; d++) {
      points_batch_factors[d] = 1;
      for (int j = d + 1; j < batch_rank; j++) {
        points_batch_factors[d] *= points_batch_dims[j];
      }
    }

    // Obtain pointers to non-uniform data c and Fourier mode
    // coefficients f.
    Complex<Device, FloatType>* c = nullptr;
    Complex<Device, FloatType>* f = nullptr;
    int source_index;
    int target_index;
    int* c_index;
    int* f_index;
    switch (type) {
      case TransformType::TYPE_1:  // nonuniform to uniform
        c = source;
        f = target;
        c_index = &source_index;
        f_index = &target_index;
        break;
      case TransformType::TYPE_2:  // uniform to nonuniform
        c = target;
        f = source;
        c_index = &target_index;
        f_index = &source_index;
        break;
    }

    // NUFFT options.
    Options options;

    if (op_type != OpType::NUFFT) {
      options.spread_only = true;
      options.upsampling_factor = 2.0;
    }

    // Intra-op threading.
    const DeviceBase::CpuWorkerThreads& worker_threads =
        *ctx->device()->tensorflow_cpu_worker_threads();
    options.num_threads = worker_threads.num_threads;

    // Make inlined vector from pointer to number of modes. TODO: use inlined
    // vector for all of num_modes.
    int num_modes_int[3] = {1, 1, 1};
    for (int i = 0; i < rank; ++i) {
      num_modes_int[i] = static_cast<int>(num_modes[i]);
    }

    // Make the NUFFT plan.
    auto plan = std::make_unique<Plan<Device, FloatType>>(ctx);
    TF_RETURN_IF_ERROR(plan->initialize(
        type, rank, num_modes_int, fft_direction,
        num_transforms, tol, max_batch_size, options));

    // Pointers to a certain batch.
    Complex<Device, FloatType>* c_batch = nullptr;
    Complex<Device, FloatType>* f_batch = nullptr;
    FloatType* points_batch = nullptr;

    FloatType* points_x = nullptr;
    FloatType* points_y = nullptr;
    FloatType* points_z = nullptr;

    gtl::InlinedVector<int, 8> source_batch_indices(batch_rank);

    for (int call_index = 0; call_index < num_calls; call_index++) {
      points_batch = points + call_index * num_points * rank;
      switch (rank) {
        case 1:
          points_x = points_batch;
          break;
        case 2:
          points_x = points_batch;
          points_y = points_batch + num_points;
          break;
        case 3:
          points_x = points_batch;
          points_y = points_batch + num_points;
          points_z = points_batch + num_points * 2;
          break;
      }

      // Set the point coordinates.
      TF_RETURN_IF_ERROR(plan->set_points(
          num_points, points_x, points_y, points_z));

      // Compute indices.
      source_index = 0;
      target_index = call_index;
      int temp_index = call_index;
      for (int d = 0; d < batch_rank; d++) {
        source_batch_indices[d] = temp_index / points_batch_factors[d];
        temp_index %= points_batch_factors[d];
        if (source_batch_dims[d] == 1) {
          source_batch_indices[d] = 0;
        }
        source_index += source_batch_indices[d] * source_batch_factors[d];
      }

      c_batch = c + *c_index * num_transforms * num_points;
      f_batch = f + *f_index * num_transforms * num_coeffs;

      // Execute the NUFFT.
      switch (op_type) {
        case OpType::NUFFT:
          TF_RETURN_IF_ERROR(plan->execute(c_batch, f_batch));
          break;
        case OpType::INTERP:
          TF_RETURN_IF_ERROR(plan->interp(c_batch, f_batch));
          break;
        case OpType::SPREAD:
          TF_RETURN_IF_ERROR(plan->spread(c_batch, f_batch));
          break;
      }
    }
    return Status::OK();
  }
};

template<typename Device, typename FloatType>
struct DoNUFFT : DoNUFFTBase<Device, FloatType> {
  Status operator()(OpKernelContext* ctx,
                    TransformType type,
                    int rank,
                    FftDirection fft_direction,
                    int num_transforms,
                    FloatType tol,
                    int max_batch_size,
                    OpType op_type,
                    int64_t batch_rank,
                    int64_t* source_batch_dims,
                    int64_t* points_batch_dims,
                    int64_t* num_modes,
                    int64_t num_points,
                    FloatType* points,
                    Complex<Device, FloatType>* source,
                    Complex<Device, FloatType>* target);
};

}  // namespace nufft
}  // namespace tensorflow

#endif  // TENSORFLOW_NUFFT_CC_KERNELS_NUFFT_KERNELS_H_
