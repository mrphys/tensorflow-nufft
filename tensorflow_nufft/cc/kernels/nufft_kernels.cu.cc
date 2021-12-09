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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"

#include "tensorflow_nufft/cc/kernels/finufft/gpu/cufinufft.h"
#include "tensorflow_nufft/cc/kernels/nufft_options.h"
#include "tensorflow_nufft/cc/kernels/nufft_plan.h"
#include "tensorflow_nufft/cc/kernels/nufft_kernels.h"


namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace nufft {

template<>
int setpts<GPUDevice, float>(
    Plan<GPUDevice, float>* plan,
    int64_t M, float* x, float* y, float* z,
    int64_t N, float* s, float* t, float* u) {
  plan->set_points(M, x, y, z);
  return 0;
};

template<>
int setpts<GPUDevice, double>(
    Plan<GPUDevice, double>* plan,
    int64_t M, double* x, double* y, double* z,
    int64_t N, double* s, double* t, double* u) {
  plan->set_points(M, x, y, z);
  return 0;
};

template<>
int execute<GPUDevice, float>(
    Plan<GPUDevice, float>* plan,
    std::complex<float>* c, std::complex<float>* f) {
  return cufinufftf_execute(
    reinterpret_cast<cuFloatComplex*>(c),
    reinterpret_cast<cuFloatComplex*>(f),
    plan);
};

template<>
int execute<GPUDevice, double>(
    Plan<GPUDevice, double>* plan,
    std::complex<double>* c, std::complex<double>* f) {
  return cufinufft_execute(
    reinterpret_cast<cuDoubleComplex*>(c),
    reinterpret_cast<cuDoubleComplex*>(f),
    plan);
};

template<>
int interp<GPUDevice, float>(
    Plan<GPUDevice, float>* plan,
    std::complex<float>* c, std::complex<float>* f) {
  return cufinufftf_interp(
    reinterpret_cast<cuFloatComplex*>(c),
    reinterpret_cast<cuFloatComplex*>(f),
    plan);
};

template<>
int interp<GPUDevice, double>(
    Plan<GPUDevice, double>* plan,
    std::complex<double>* c, std::complex<double>* f) {
  return cufinufft_interp(
    reinterpret_cast<cuDoubleComplex*>(c),
    reinterpret_cast<cuDoubleComplex*>(f),
    plan);
};

template<>
int spread<GPUDevice, float>(
    Plan<GPUDevice, float>* plan,
    std::complex<float>* c, std::complex<float>* f) {
  return cufinufftf_spread(
    reinterpret_cast<cuFloatComplex*>(c),
    reinterpret_cast<cuFloatComplex*>(f),
    plan);
};

template<>
int spread<GPUDevice, double>(
    Plan<GPUDevice, double>* plan,
    std::complex<double>* c, std::complex<double>* f) {
  return cufinufft_spread(
    reinterpret_cast<cuDoubleComplex*>(c),
    reinterpret_cast<cuDoubleComplex*>(f),
    plan);
};

}   // namespace nufft

using namespace tensorflow::nufft;

template<typename T>
struct DoNUFFT<GPUDevice, T> : DoNUFFTBase<GPUDevice, T> {
  Status operator()(OpKernelContext* ctx,
                    TransformType type,
                    int rank,
                    FftDirection fft_direction,
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
    return this->compute(
      ctx, type, rank, fft_direction, num_transforms, tol, optype,
      nbdims, source_bdims, points_bdims,
      nmodes, npts, points, source, target);
  }
};

// Explicitly instantiate functors for the types of OpKernels registered.
template struct DoNUFFT<GPUDevice, float>;
template struct DoNUFFT<GPUDevice, double>;

}   // namespace tensorflow

#endif  // GOOGLE_CUDA
