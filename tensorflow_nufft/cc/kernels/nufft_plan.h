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

/* Copyright 2017-2021 The Simons Foundation. All Rights Reserved.

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

#ifndef TENSORFLOW_NUFFT_KERNELS_NUFFT_PLAN_H_
#define TENSORFLOW_NUFFT_KERNELS_NUFFT_PLAN_H_

#define EIGEN_USE_THREADS
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif // GOOGLE_CUDA

#include <cstdint>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuComplex.h"
#include "third_party/gpus/cuda/include/cufft.h"
#endif
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow_nufft/cc/kernels/fftw_api.h"
#include "tensorflow_nufft/cc/kernels/nufft_options.h"
#include "tensorflow_nufft/cc/kernels/nufft_spread.h"


namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace nufft {

// The maximum allowed array size.
constexpr static int kMaxArraySize = 2000000000; // 2 billion points

// Max number of positive quadrature nodes for kernel FT.
constexpr static int kMaxQuadNodes = 100;

// Largest possible kernel spread width per dimension, in fine grid points.
constexpr static int kMaxKernelWidth = 16;

// Mathematical constants.
template<typename FloatType>
constexpr static FloatType kPi = FloatType(3.14159265358979329);

template<typename FloatType>
constexpr static std::complex<FloatType> kImaginaryUnit = std::complex<FloatType>(0.0, 1.0);

template<typename FloatType>
constexpr FloatType kEpsilon;
template<>
constexpr float kEpsilon<float> = 6e-08f;
template<>
constexpr double kEpsilon<double> = 1.1e-16;

template<typename Device, typename FloatType>
struct ComplexType;

template<>
struct ComplexType<CPUDevice, float> {
  using Type = std::complex<float>;
};

template<>
struct ComplexType<CPUDevice, double> {
  using Type = std::complex<double>;
};

#ifdef GOOGLE_CUDA
template<>
struct ComplexType<GPUDevice, float> {
  using Type = cuFloatComplex;
};

template<>
struct ComplexType<GPUDevice, double> {
  using Type = cuDoubleComplex;
};
#endif

// Transform type naming by:
// Dutt A, Rokhlin V. Fast Fourier transforms for nonequispaced data. SIAM
// Journal on Scientific computing. 1993 Nov;14(6):1368-93.
enum class TransformType {
  TYPE_1, // non-uniform to uniform
  TYPE_2, // uniform to non-uniform
  TYPE_3  // non-uniform to non-uniform (not implemented)
};

// Direction of the FFT. The enum value is the sign of the exponent.
enum class FftDirection {
  FORWARD = -1,
  BACKWARD = 1
};

template<typename Device, typename FloatType>
class PlanBase {

 public:
  PlanBase(OpKernelContext* context)
      : device_(context->eigen_device<Device>()) { }

 public: // TODO: make protected

  // The type of the transform. See enum above.
  TransformType type_;

  // The rank of the transform (number of dimensions). Must be 1, 2 or 3.
  int rank_;

  // The number of modes in each dimension.
  gtl::InlinedVector<int, 4> num_modes_;

  // Direction of the FFT. See enum above.
  FftDirection fft_direction_;

  // How many transforms to compute in one go.
  int num_transforms_;

  // Advanced NUFFT options.
  Options options_;

  // Size of the fine grid.
  gtl::InlinedVector<int, 4> grid_sizes_;

  // Total number of modes. The product of the elements of num_modes_.
  int64_t num_modes_total_;

  // Total number of points in the fine grid. The product of the elements of
  // grid_sizes_.
  int64_t num_grid_points_;

  // Reference to the active device.
  const Device& device_;
};

template<typename Device, typename FloatType>
class Plan;

template<typename FloatType>
class Plan<CPUDevice, FloatType> : public PlanBase<CPUDevice, FloatType> {

 public:

  // The main data type this plan operates with; either complex float or double.
  using DType = typename ComplexType<CPUDevice, FloatType>::Type;

  // The corresponding FFTW type.
  using FftwType = typename fftw::ComplexType<FloatType>::Type;

  // Creates a new NUFFT plan for the CPU. Allocates memory for internal working
  // arrays, evaluates spreading kernel coefficients, and instantiates the FFT
  // plan.
  Plan(OpKernelContext* context,
       TransformType type,
       int rank,
       gtl::InlinedVector<int, 4> num_modes,
       FftDirection fft_direction,
       int num_transforms,
       FloatType tol,
       const Options& options);

  // Frees any dynamically allocated memory not handled by the op kernel
  // context, destroys the FFT plan and cleans up persistent FFTW data such as
  // accumulated wisdom.
  ~Plan();

 public: // TODO: make protected.

  // Number of computations in one batch.
  int batch_size_;

  // Number of batches in one execution (includes all the transforms in
  // num_transforms_).
  int num_batches_;

  // Batch of fine grids for FFTW to plan and execute. This is usually the
  // largest array allocated by NUFFT.
  Tensor fine_grid_;

  // A convenience pointer to the fine grid array for FFTW calls.
  FftwType* fine_grid_data_;

  // Relative user tol.
  FloatType tol_;

  // The FFTW plan for FFTs.
  typename fftw::PlanType<FloatType>::Type fft_plan_;

  // The parameters for the spreading algorithm/s.
  SpreadParameters<FloatType> spread_params_;

  int nj;          // number of NU pts in type 1,2 (for type 3, num input x pts)
  int nk;          // number of NU freq pts (type 3 only)

  // int64_t nf1;      // size of internal fine grid in x (1) direction
  // int64_t nf2;      // " y
  // int64_t nf3;      // " z
  int64_t nf;       // total # fine grid points (product of the above three)
  
  FloatType* phiHat1;    // FT of kernel in t1,2, on x-axis mode grid
  FloatType* phiHat2;    // " y-axis.
  FloatType* phiHat3;    // " z-axis.
  
  int64_t *sortIndices;  // precomputed NU pt permutation, speeds spread/interp
  bool didSort;         // whether binsorting used (false: identity perm used)

  FloatType *X, *Y, *Z;  // for t1,2: ptr to user-supplied NU pts (no new allocs).
                   // for t3: allocated as "primed" (scaled) src pts x'_j, etc

  // type 3 specific
  FloatType *S, *T, *U;  // pointers to user's target NU pts arrays (no new allocs)
  FloatType *Sp, *Tp, *Up;  // internal primed targs (s'_k, etc), allocated
};

#if GOOGLE_CUDA
template<typename FloatType>
class Plan<GPUDevice, FloatType> : public PlanBase<GPUDevice, FloatType> {

 public:

  // The main data type this plan operates with; either complex float or double.
  using DType = typename ComplexType<GPUDevice, FloatType>::Type;

  Plan(OpKernelContext* context,
       TransformType type,
       int rank,
       gtl::InlinedVector<int, 4> num_modes,
       FftDirection fft_direction,
       int num_transforms,
       FloatType tol,
       const Options& options);

  // Frees any dynamically allocated memory not handled by the op kernel and
  // destroys the FFT plan.
  ~Plan();

  // Sets the number and coordinates of the non-uniform points. Allocates GPU
  // arrays with the required sizes. Rescales the non-uniform points to the
  // range used by the spreader. Determines the spreader parameters that depend
  // on the non-uniform points.
  Status set_points(int num_points,
                    FloatType* points_x,
                    FloatType* points_y,
                    FloatType* points_z);

 public:

  // Batch of fine grids for cuFFT to plan and execute. This is usually the
  // largest array allocated by NUFFT.
  Tensor fine_grid_;

  // A convenience pointer to the fine grid array.
  DType* fine_grid_data_;

  // Tensors in device memory. Used for deconvolution. Empty in spread/interp
  // mode. Only the first `rank` tensors are allocated.
  Tensor kernel_fseries_[3];

  // Convenience raw pointers to above tensors. These are device pointers. Only
  // the first `rank` pointers are valid.
  FloatType* kernel_fseries_data_[3];

  // The cuFFT plan.
  cufftHandle fft_plan_;

  // The parameters for the spreading algorithm/s.
  SpreadParameters<FloatType> spread_params_;
  
  
  // std::unique_ptr<Spreader<GPUDevice, FloatType>> spreader_; // owned

  int M;
  int nf1;
  int nf2;
  int nf3;
  int ms;
  int mt;
  int mu;

  int totalnumsubprob;
  int byte_now;
  
  FloatType *kx;
  FloatType *ky;
  FloatType *kz;
  typename ComplexType<GPUDevice, FloatType>::Type* c;
  typename ComplexType<GPUDevice, FloatType>::Type* fk;

  // Arrays that used in subprob method
  int *idxnupts;//length: #nupts, index of the nupts in the bin-sorted order
  int *sortidx; //length: #nupts, order inside the bin the nupt belongs to
  int *numsubprob; //length: #bins,  number of subproblems in each bin
  int *binsize; //length: #bins, number of nonuniform ponits in each bin
  int *binstartpts; //length: #bins, exclusive scan of array binsize
  int *subprob_to_bin;//length: #subproblems, the bin the subproblem works on 
  int *subprobstartpts;//length: #bins, exclusive scan of array numsubprob

  // Extra arrays for Paul's method
  int *finegridsize;
  int *fgstartpts;

  // Arrays for 3d (need to sort out)
  int *numnupts;
  int *subprob_to_nupts;
  
};
#endif // GOOGLE_CUDA

} // namespace nufft
} // namespace tensorflow

#endif // TENSORFLOW_NUFFT_KERNELS_NUFFT_PLAN_H_
