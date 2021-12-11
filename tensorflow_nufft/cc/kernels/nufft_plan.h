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
constexpr static FloatType kOneOverTwoPi = FloatType(0.159154943091895336);

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

// Specifies the direction of spreading.
enum class SpreadDirection {
  SPREAD, // non-uniform to uniform
  INTERP  // uniform to non-uniform
};

template<typename FloatType>
struct SpreadParameters {

  // The spread direction (U->NU or NU->U). See enum above.
  SpreadDirection spread_direction;

  // Whether to sort the non-uniform points.
  SortPoints sort_points = SortPoints::AUTO;

  // Specifies the spread method.
  SpreadMethod spread_method = SpreadMethod::AUTO;

  // If true, do only spreading/interpolation step
  // (no FFT or amplification/deconvolution).
  bool spread_only;

  // TODO: revise the following options.
  int pirange;            // 0: NU periodic domain is [0,N), 1: domain [-pi,pi)
  bool check_bounds;      // 0: don't check NU pts in 3-period range; 1: do
  int kerevalmeth;        // 0: direct exp(sqrt()), or 1: Horner ppval, fastest
  bool pad_kernel;            // 0: no pad w to mult of 4, 1: do pad
                          // (this helps SIMD for kerevalmeth=0, eg on i7).
  int num_threads;        // # threads for spreadinterp (0: use max avail)
  int sort_threads;       // # threads for sort (0: auto-choice up to num_threads)
  int max_subproblem_size; // # pts per t1 subprob; sets extra RAM per thread
  int flags;              // binary flags for timing only (may give wrong ans
                          // if changed from 0!). See spreadinterp.h
  int verbosity;          // 0: silent, 1: small text output, 2: verbose
  int atomic_threshold;   // num threads before switching spreadSorted to using atomic ops
  double upsampling_factor;       // sigma, upsampling factor

  // Parameters of the "exponential of semicircle" spreading kernel.
  int kernel_width;
  FloatType kernel_beta;
  FloatType kernel_half_width;
  FloatType kernel_c;
  FloatType kernel_scale;

  #if GOOGLE_CUDA
  // Used for 3D subproblem method. 0 means automatic selection.
  dim3 gpu_bin_size = {0, 0, 0};

  // Used for 3D spread-block-gather method. 0 means automatic selection.
  dim3 gpu_obin_size = {0, 0, 0};
  #endif // GOOGLE_CUDA
};

template<typename Device, typename FloatType>
class PlanBase {

 public:
  // The main data type this plan operates with; either complex float or double.
  using DType = typename ComplexType<Device, FloatType>::Type;

  PlanBase(OpKernelContext* context)
      : device_(context->eigen_device<Device>()) { }

  virtual Status set_points(int num_points,
                            FloatType* points_x,
                            FloatType* points_y,
                            FloatType* points_z) = 0;

  virtual Status execute(DType* c, DType* f) = 0;

  virtual Status interp(DType* c, DType* f) = 0;

  virtual Status spread(DType* c, DType* f) = 0;

 public: // TODO: make protected

  // The type of the transform. See enum above.
  TransformType type_;

  // The rank of the transform (number of dimensions). Must be 1, 2 or 3.
  int rank_;

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

  Status set_points(int num_points,
                    FloatType* points_x,
                    FloatType* points_y,
                    FloatType* points_z) override;

  Status execute(DType* c, DType* f) override;

  Status interp(DType* c, DType* f) override;

  Status spread(DType* c, DType* f) override;

 public: // TODO: make private.

  // Number of computations in one batch.
  int batch_size_;

  // Number of batches in one execution (includes all the transforms in
  // num_transforms_).
  int num_batches_;

  // Batch of fine grids for FFTW to plan and execute. This is usually the
  // largest array allocated by NUFFT.
  Tensor grid_tensor_;

  // A convenience pointer to the fine grid array for FFTW calls.
  FftwType* grid_data_;

  // Relative user tol.
  FloatType tol_;

  // The FFTW plan for FFTs.
  typename fftw::PlanType<FloatType>::Type fft_plan_;

  // The parameters for the spreading algorithm/s.
  SpreadParameters<FloatType> spread_params_;

  // The number of modes in each dimension.
  gtl::InlinedVector<int, 4> num_modes_;

  int nj;          // number of NU pts in type 1,2 (for type 3, num input x pts)
  int nk;          // number of NU freq pts (type 3 only)

  int64_t nf;       // total # fine grid points (product of the above three)
  
  FloatType* phiHat1;    // FT of kernel in t1,2, on x-axis mode grid
  FloatType* phiHat2;    // " y-axis.
  FloatType* phiHat3;    // " z-axis.
  
  int64_t *sortIndices;  // precomputed NU pt permutation, speeds spread/interp
  bool didSort;         // whether binsorting used (false: identity perm used)

  FloatType *X, *Y, *Z;  // for t1,2: ptr to user-supplied NU pts (no new allocs).
                   // for t3: allocated as "primed" (scaled) src pts x'_j, etc
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
                    FloatType* points_z) override;

  /*
	"exec" stage (single and double precision versions).

	The actual transformation is done here. Type and dimension of the
	transformation are defined in d_plan in previous stages.

        See ../docs/cppdoc.md for main user-facing documentation.

	Input/Output:
	d_c   a size d_plan->num_points_ CPX array on gpu (input for Type 1; output for Type
	      2)
	d_fk  a size d_plan->ms*d_plan->mt*d_plan->mu CPX array on gpu ((input for
	      Type 2; output for Type 1)

	Notes:
        i) Here CPX is a defined type meaning either complex<float> or complex<double>
	    to match the precision of the library called.
        ii) All operations are done on the GPU device (hence the d_* names)

	Melody Shih 07/25/19; Barnett 2/16/21.
  */
  Status execute(DType* d_c, DType* d_fk) override;

  Status interp(DType* d_c, DType* d_fk) override;

  Status spread(DType* d_c, DType* d_fk) override;

 private:
  
  Status init_spreader();

  Status init_spreader_nupts_driven();

  Status init_spreader_subproblem();

  /*  
    2D Type-1 NUFFT

    This function is called in "exec" stage (See ../cufinufft.cu).
    It includes (copied from doc in finufft library)
      Step 1: spread data to oversampled regular mesh using kernel
      Step 2: compute FFT on uniform mesh
      Step 3: deconvolve by division of each Fourier mode independently by the
              Fourier series coefficient of the kernel.

    Melody Shih 07/25/19		
  */
  Status execute_type_1(DType* d_c, DType* d_fk);

  /*  
    2D Type-2 NUFFT

    This function is called in "exec" stage (See ../cufinufft.cu).
    It includes (copied from doc in finufft library)
      Step 1: deconvolve (amplify) each Fourier mode, dividing by kernel 
              Fourier coeff
      Step 2: compute FFT on uniform mesh
      Step 3: interpolate data to regular mesh

    Melody Shih 07/25/19
  */
  Status execute_type_2(DType* d_c, DType* d_fk);

  Status spread_batch(int blksize);

  Status interp_batch(int blksize);

  Status spread_batch_nupts_driven(int blksize);

  Status spread_batch_subproblem(int blksize);

  Status interp_batch_nupts_driven(int blksize);

  Status interp_batch_subproblem(int blksize);

  /* 
  wrapper for deconvolution & amplication in 2D.

  Melody Shih 07/25/19
  */
  Status deconvolve_batch(int blksize);

 public:

  // Batch of fine grids for cuFFT to plan and execute. This is usually the
  // largest array allocated by NUFFT.
  Tensor grid_tensor_;

  // A convenience pointer to the fine grid array.
  DType* grid_data_;

  // Tensors in device memory. Used for deconvolution. Empty in spread/interp
  // mode. Only the first `rank` tensors are allocated.
  Tensor fseries_tensor_[3];

  // Convenience raw pointers to above tensors. These are device pointers. Only
  // the first `rank` pointers are valid.
  FloatType* fseries_data_[3];

  // The cuFFT plan.
  cufftHandle fft_plan_;

  // The parameters for the spreading algorithm/s.
  SpreadParameters<FloatType> spread_params_;
  
  int grid_dims_[3];
  int grid_size_;

  FloatType* points_[3];
  int num_points_;

  int bin_dims_[3];
  int num_bins_[3];
  int bin_count_;

  int num_modes_[3];
  int mode_count_;

  int nf1;
  int nf2;
  int nf3;
  int ms;
  int mt;
  int mu;

  
  
  // Internal pointer to non-uniform data.
  DType* c_;

  // Internal pointer to uniform data.
  DType* f_;

  // Total number of subproblems.
  int subprob_count_;
  // Indices of the non-uniform points in the bin-sorted order. When allocated,
  // it has length equal to num_points_. This is a device pointer.
  int* idx_nupts_;
  // Sort indices of non-uniform points within their corresponding bins. When
  // allocated, it has length equal to num_points_. This is a device pointer.
  int* sort_idx_;
  // Number of subproblems in each bin. When allocated, it has length equal to
  // bin_count_. This is a device pointer.
  int* num_subprob_;
  // Number of non-uniform points in each bin. When allocated, it has length
  // equal to bin_count_. This is a device pointer.
  int* bin_sizes_;
  // The start points for each bin. When allocated, it has length equal to
  // bin_count_. This is a device pointer.
  int* bin_start_pts_;
  // The bin each subproblem works on. When allocated, it has length equal to
  // the number of subproblems. This is a device pointer.
  int* subprob_bins_;
  // The start points for each subproblem. When allocated, it has length equal
  // to the bin_count_. This is a device pointer.
  int* subprob_start_pts_;
};
#endif // GOOGLE_CUDA

}  // namespace nufft
}  // namespace tensorflow

#endif // TENSORFLOW_NUFFT_KERNELS_NUFFT_PLAN_H_
