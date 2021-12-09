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

#ifndef TENSORFLOW_NUFFT_KERNELS_NUFFT_KERNELS_H_
#define TENSORFLOW_NUFFT_KERNELS_NUFFT_KERNELS_H_

#include <complex>
#include <cstdint>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/status.h"

#include "tensorflow_nufft/cc/kernels/nufft_options.h"
#include "tensorflow_nufft/cc/kernels/nufft_plan.h"


namespace tensorflow {
namespace nufft {

template<typename Device, typename T>
int setpts(
    Plan<Device, T>* plan,
    int64_t M, T* x, T* y, T* z,
    int64_t N, T* s, T* t, T* u);

template<typename Device, typename T>
int execute(
    Plan<Device, T>* plan,
    std::complex<T>* c, std::complex<T>* f);

template<typename Device, typename T>
int interp(
    Plan<Device, T>* plan,
    std::complex<T>* c, std::complex<T>* f);

template<typename Device, typename T>
int spread(
    Plan<Device, T>* plan,
    std::complex<T>* c, std::complex<T>* f);

}   // namespace nufft

enum class OpType { NUFFT, INTERP, SPREAD };

template<typename Device, typename T>
struct DoNUFFTBase {

  Status compute(OpKernelContext* ctx,
                 nufft::TransformType type,
                 int rank,
                 nufft::FftDirection fft_direction,
                 int num_transforms,
                 T tol,
                 OpType optype,
                 int64_t nbdims,
                 int64_t* source_bdims,
                 int64_t* points_bdims,
                 int64_t* nmodes,
                 int64_t npts,
                 T* points,
                 std::complex<T>* source,
                 std::complex<T>* target) {

    // Number of coefficients.
    int ncoeffs = 1;
    for (int d = 0; d < rank; d++) {
      ncoeffs *= nmodes[d];
    }

    // Number of calls to FINUFFT execute.
    int ncalls = 1;
    for (int d = 0; d < nbdims; d++) {
      ncalls *= points_bdims[d];
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
    std::complex<T>* strengths = nullptr;
    std::complex<T>* coeffs = nullptr;
    int csrc;
    int ctgt;
    int* pcs;
    int* pcc;
    switch (type) {
      case nufft::TransformType::TYPE_1: // nonuniform to uniform
        strengths = source;
        coeffs = target;
        pcs = &csrc;
        pcc = &ctgt;
        break;
      case nufft::TransformType::TYPE_2: // uniform to nonuniform
        strengths = target;
        coeffs = source;
        pcs = &ctgt;
        pcc = &csrc;
        break;
    }

    // NUFFT options.
    nufft::Options options;

    if (optype != OpType::NUFFT) {
      options.spread_only = true;
      options.upsampling_factor = 2.0;
    }

    // Intra-op threading.
    const DeviceBase::CpuWorkerThreads& worker_threads =
        *ctx->device()->tensorflow_cpu_worker_threads();
    options.num_threads = worker_threads.num_threads;

    // Make inlined vector from pointer to number of modes. TODO: use inlined
    // vector for all of num_modes.
    gtl::InlinedVector<int, 4> num_modes(rank);
    for (int i = 0; i < rank; ++i) {
      num_modes[i] = nmodes[i];
    }

    // Make the NUFFT plan.
    auto plan = std::make_unique<nufft::Plan<Device, T>>(
        ctx, type, rank, num_modes, fft_direction,
        num_transforms, tol, options);

    int err;

    // Pointers to a certain batch.
    std::complex<T>* bstrengths = NULL;
    std::complex<T>* bcoeffs = NULL;
    T* bpoints = NULL;

    T* points_x = NULL;
    T* points_y = NULL;
    T* points_z = NULL;

    gtl::InlinedVector<int, 8> source_binds(nbdims);

    for (int c = 0; c < ncalls; c++) {
      
      bpoints = points + c * npts * rank;

      switch (rank) {
        case 1:
          points_x = bpoints;
          break;
        case 2:
          points_x = bpoints;
          points_y = bpoints + npts;
          break;
        case 3:
          points_x = bpoints;
          points_y = bpoints + npts;
          points_z = bpoints + npts * 2;
          break;
      }
      
      // Set the point coordinates.
      err = nufft::setpts<Device, T>(
          plan.get(),
          npts, points_x, points_y, points_z,
          0, NULL, NULL, NULL);
        
      if (err > 0) {
        return errors::Internal("Failed during `nufft::setpts`: ", err);
      }

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

      bstrengths = strengths + *pcs * num_transforms * npts;
      bcoeffs = coeffs + *pcc * num_transforms * ncoeffs;

      // Execute the NUFFT.
      switch (optype)
      {
      case OpType::NUFFT:
        err = nufft::execute<Device, T>(plan.get(), bstrengths, bcoeffs);
        if (err > 0) {
          return errors::Internal("Failed during `nufft::execute`: ", err);
        }
        break;
      case OpType::INTERP:
        err = nufft::interp<Device, T>(plan.get(), bstrengths, bcoeffs);
        if (err > 0) {
          return errors::Internal("Failed during `nufft::interp`: ", err);
        }
        break;
      case OpType::SPREAD:
        err = nufft::spread<Device, T>(plan.get(), bstrengths, bcoeffs);
        if (err > 0) {
          return errors::Internal("Failed during `nufft::spread`: ", err);
        }
        break;
      }
    }

    return Status::OK();
  }
};

template<typename Device, typename T>
struct DoNUFFT : DoNUFFTBase<Device, T> {

  Status operator()(OpKernelContext* ctx,
                    nufft::TransformType type,
                    int rank,
                    nufft::FftDirection fft_direction,
                    int num_transforms,
                    T tol,
                    OpType optype,
                    int64_t nbdims,
                    int64_t* source_bdims,
                    int64_t* points_bdims,
                    int64_t* nmodes,
                    int64_t npts,
                    T* points,
                    std::complex<T>* source,
                    std::complex<T>* target);
};

}   // namespace tensorflow

#endif // TENSORFLOW_NUFFT_KERNELS_NUFFT_KERNELS_H_