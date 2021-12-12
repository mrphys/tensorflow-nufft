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
constexpr static int kMaxArraySize = 2000000000;  // 2 billion points

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
      : context_(context),
        device_(context->eigen_device<Device>()) { }

  // Frees any dynamically allocated memory not handled by the op kernel
  // context, destroys the FFT plan and cleans up persistent FFTW data such as
  // accumulated wisdom.
  virtual ~PlanBase() { }

  // Initializes a new NUFFT plan. Allocates memory for internal working arrays,
  // evaluates spreading kernel coefficients, and instantiates the FFT plan.
  virtual Status initialize(TransformType type,
                            int rank,
                            int* num_modes,
                            FftDirection fft_direction,
                            int num_transforms,
                            FloatType tol,
                            const Options& options) = 0;

  // Sets the number and coordinates of the non-uniform points. Allocates arrays
  // arrays that depend on the number of points. Maybe sorts the non-uniform
  // points. Maybe rescales the non-uniform points to the range used by the
  // spreader. Determines the spreader parameters that depend on the non-uniform
  // points. Must be called after `initialize`.
  virtual Status set_points(int num_points,
                            FloatType* points_x,
                            FloatType* points_y,
                            FloatType* points_z) = 0;

  // Executes the plan. Must be called after initialize() and set_points(). `c`
  // and `f` are the non-uniform and uniform grid arrays, respectively. Each
  // may be an input or an output depending on the type of the transform.
  virtual Status execute(DType* c, DType* f) = 0;

  // Performs the interpolation step only. Must be called after initialize() and
  // set_points().
  virtual Status interp(DType* c, DType* f) = 0;

  // Performs the spreading step only. Must be called after initialize() and
  // set_points().
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
  // The number of modes along each dimension.
  int num_modes_[3];
  // The total number of modes.
  int mode_count_;
  // The fine grid's dimension sizes. Unused dimensions are set to 1.
  int grid_dims_[3];
  // The total element count of the fine grid.
  int grid_size_;
  // Pointers to the non-uniform point coordinates. In the GPU case, these are
  // device pointers. These pointers are not owned by the plan. Unused pointers
  // are set to nullptr.
  FloatType* points_[3];
  // The total number of points.
  int num_points_;
  // Pointer to the op kernel context.
  OpKernelContext* context_;
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

  Plan(OpKernelContext* context)
      : PlanBase<CPUDevice, FloatType>(context) { }

  ~Plan();

  Status initialize(TransformType type,
                    int rank,
                    int* num_modes,
                    FftDirection fft_direction,
                    int num_transforms,
                    FloatType tol,
                    const Options& options) override;

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
  // Tensors in host memory. Used for deconvolution. Empty in spread/interp
  // mode. Only the first `rank` tensors are allocated.
  Tensor fseries_tensor_[3];
  // Convenience raw pointers to above tensors. Only the first `rank` pointers
  // are valid.
  FloatType* fseries_data_[3];
  // Precomputed non-uniform point permutation, used to speed up spread/interp.
  int64_t* sort_indices_;
  // Whether bin-sorting was used.
  bool did_sort_;
};

#if GOOGLE_CUDA
template<typename FloatType>
class Plan<GPUDevice, FloatType> : public PlanBase<GPUDevice, FloatType> {

 public:
  // The main data type this plan operates with; either complex float or double.
  using DType = typename ComplexType<GPUDevice, FloatType>::Type;

  Plan(OpKernelContext* context)
      : PlanBase<GPUDevice, FloatType>(context) { }

  ~Plan();

  Status initialize(TransformType type,
                    int rank,
                    int* num_modes,
                    FftDirection fft_direction,
                    int num_transforms,
                    FloatType tol,
                    const Options& options) override;
  
  Status set_points(int num_points,
                    FloatType* points_x,
                    FloatType* points_y,
                    FloatType* points_z) override;
  
  Status execute(DType* d_c, DType* d_fk) override;
  
  Status interp(DType* d_c, DType* d_fk) override;

  Status spread(DType* d_c, DType* d_fk) override;

 private:
  Status init_spreader();
  Status init_spreader_nupts_driven();
  Status init_spreader_subproblem();
  // Performs type-1 NUFFT. This consists of 3 steps: (1) spreading of
  // non-uniform data to fine grid, (2) FFT on fine grid, and (3) deconvolution
  // (division of modes by Fourier series of kernel).
  Status execute_type_1(DType* d_c, DType* d_fk);
  // Performs type-2 NUFFT. This consists of 3 steps: (1) deconvolution
  // (division of modes by Fourier series of kernel), (2) FFT on fine grid, and
  // (3) interpolation of data to non-uniform points.
  Status execute_type_2(DType* d_c, DType* d_fk);
  Status spread_batch(int batch_size);
  Status interp_batch(int batch_size);
  Status spread_batch_nupts_driven(int batch_size);
  Status spread_batch_subproblem(int batch_size);
  Status interp_batch_nupts_driven(int batch_size);
  Status interp_batch_subproblem(int batch_size);
  // Deconvolve and/or amplify a batch of data.
  Status deconvolve_batch(int batch_size);
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
  // The GPU bin dimension sizes.
  int bin_dims_[3];
  // The number of GPU bins.
  int num_bins_[3];
  // The total bin count.
  int bin_count_;
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
