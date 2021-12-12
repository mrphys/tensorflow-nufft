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
                 OpType op_type,
                 int64_t nbdims,
                 int64_t* source_bdims,
                 int64_t* points_bdims,
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
    for (int d = 0; d < nbdims; d++) {
      num_calls *= points_bdims[d];
    }

    // Factors to transform linear indices to subindices and viceversa.
    gtl::InlinedVector<int, 8> source_bfactors(nbdims);
    for (int d = 0; d < nbdims; d++) {
      source_bfactors[d] = 1;
      for (int j = d + 1; j < nbdims; j++) {
        source_bfactors[d] *= source_bdims[j];
      }
    }

    gtl::InlinedVector<int, 8> points_bfactors(nbdims);
    for (int d = 0; d < nbdims; d++) {
      points_bfactors[d] = 1;
      for (int j = d + 1; j < nbdims; j++) {
        points_bfactors[d] *= points_bdims[j];
      }
    }
    
    // Obtain pointers to non-uniform strengths and Fourier mode
    // coefficients.
    Complex<Device, FloatType>* strengths = nullptr;
    Complex<Device, FloatType>* coeffs = nullptr;
    int csrc;
    int ctgt;
    int* pcs;
    int* pcc;
    switch (type) {
      case TransformType::TYPE_1: // nonuniform to uniform
        strengths = source;
        coeffs = target;
        pcs = &csrc;
        pcc = &ctgt;
        break;
      case TransformType::TYPE_2: // uniform to nonuniform
        strengths = target;
        coeffs = source;
        pcs = &ctgt;
        pcc = &csrc;
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
    auto plan = std::make_unique<Plan<Device, FloatType>>(
        ctx, type, rank, num_modes_int, fft_direction,
        num_transforms, tol, options);

    // Pointers to a certain batch.
    Complex<Device, FloatType>* bstrengths = nullptr;
    Complex<Device, FloatType>* bcoeffs = nullptr;
    FloatType* points_batch = nullptr;

    FloatType* points_x = nullptr;
    FloatType* points_y = nullptr;
    FloatType* points_z = nullptr;

    gtl::InlinedVector<int, 8> source_binds(nbdims);

    for (int c = 0; c < num_calls; c++) {

      points_batch = points + c * num_points * rank;

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
      csrc = 0;
      ctgt = c;
      int ctmp = c;
      for (int d = 0; d < nbdims; d++) {
        source_binds[d] = ctmp / points_bfactors[d];
        ctmp %= points_bfactors[d];
        if (source_bdims[d] == 1) {
          source_binds[d] = 0;
        }
        csrc += source_binds[d] * source_bfactors[d];
      }

      bstrengths = strengths + *pcs * num_transforms * num_points;
      bcoeffs = coeffs + *pcc * num_transforms * num_coeffs;

      // Execute the NUFFT.
      switch (op_type) {
        case OpType::NUFFT:
          TF_RETURN_IF_ERROR(plan->execute(bstrengths, bcoeffs));
          break;
        case OpType::INTERP:
          TF_RETURN_IF_ERROR(plan->interp(bstrengths, bcoeffs));
          break;
        case OpType::SPREAD:
          TF_RETURN_IF_ERROR(plan->spread(bstrengths, bcoeffs));
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
                    OpType op_type,
                    int64_t nbdims,
                    int64_t* source_bdims,
                    int64_t* points_bdims,
                    int64_t* num_modes,
                    int64_t num_points,
                    FloatType* points,
                    Complex<Device, FloatType>* source,
                    Complex<Device, FloatType>* target);
};

}  // namespace nufft
}  // namespace tensorflow

#endif  // TENSORFLOW_NUFFT_CC_KERNELS_NUFFT_KERNELS_H_
