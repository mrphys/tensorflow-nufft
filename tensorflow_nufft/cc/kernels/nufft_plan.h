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

#ifndef TENSORFLOW_NUFFT_CC_KERNELS_NUFFT_PLAN_H_
#define TENSORFLOW_NUFFT_CC_KERNELS_NUFFT_PLAN_H_

#define EIGEN_USE_THREADS
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include <cstdint>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuComplex.h"
#include "third_party/gpus/cuda/include/cufft.h"
#include "tensorflow/core/platform/stream_executor.h"
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
constexpr static std::complex<FloatType> kImaginaryUnit =
    std::complex<FloatType>(0.0, 1.0);

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
  TYPE_1,  // non-uniform to uniform
  TYPE_2,  // uniform to non-uniform
  TYPE_3   // non-uniform to non-uniform (not implemented)
};

// Direction of the FFT. The enum value is the sign of the exponent.
enum class FftDirection {
  FORWARD = -1,
  BACKWARD = 1
};

// Specifies the direction of spreading.
enum class SpreadDirection {
  SPREAD,  // non-uniform to uniform
  INTERP   // uniform to non-uniform
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
  // Number of threads used for spreading/interpolation. If 0, use the default
  // value. Relevant for both CPU and GPU implementations.
  int num_threads;
  // Number of threads used during sorting. If 0, a default value is selected.
  // Only relevant for CPU implementation.
  int sort_threads;
  // Number of threads above which spreading is performed using atomics.
  int atomic_threshold;
  // TODO(jmontalt): revise the following options.
  int pirange;            // 0: NU periodic domain is [0,N), 1: domain [-pi,pi)
  bool check_bounds;      // 0: don't check NU pts in 3-period range; 1: do
  int kerevalmeth;        // 0: direct exp(sqrt()), or 1: Horner ppval, fastest
  bool pad_kernel;            // 0: no pad w to mult of 4, 1: do pad
                          // (this helps SIMD for kerevalmeth=0, eg on i7).
  int max_subproblem_size;  // # pts per t1 subprob; sets extra RAM per thread
  int flags;              // binary flags for timing only (may give wrong ans
                          // if changed from 0!). See spreadinterp.h
  int verbosity;          // 0: silent, 1: small text output, 2: verbose
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
  #endif  // GOOGLE_CUDA
};

template<typename Device, typename FloatType>
class PlanBase {
 public:
  // The main data type this plan operates with; either complex float or double.
  using DType = typename ComplexType<Device, FloatType>::Type;

  explicit PlanBase(OpKernelContext* context)
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

 public:  // TODO(jmontalt): make protected after refactoring FINUFFT.
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

  explicit Plan(OpKernelContext* context)
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

 protected:

  // If opts.spread_direction=1, evaluate, in the 1D case,

  //                       N1-1
  // data_nonuniform[j] =  SUM phi(kx[j] - n) data_uniform[n],   for j=0...M-1
  //                       n=0

  // If opts.spread_direction=2, evaluate its transpose, in the 1D case,

  //                   M-1
  // data_uniform[n] =  SUM phi(kx[j] - n) data_nonuniform[j],   for n=0...N1-1
  //                   j=0

  // In each case phi is the spreading kernel, which has support
  // [-opts.kernel_width/2,opts.kernel_width/2]. In 2D or 3D, the generalization with
  // product of 1D kernels is performed.
  // For 1D set N2=N3=1; for 2D set N3=1; for 3D set N1,N2,N3>1.

  // Notes:
  // No particular normalization of the spreading kernel is assumed.
  // Uniform (U) points are centered at coords
  // [0,1,...,N1-1] in 1D, analogously in 2D and 3D. They are stored in x
  // fastest, y medium, z slowest ordering, up to however many
  // dimensions are relevant; note that this is Fortran-style ordering for an
  // array f(x,y,z), but C style for f[z][y][x]. This is to match the Fortran
  // interface of the original CMCL libraries.
  // Non-uniform (NU) points kx,ky,kz are real, and may lie in the central three
  // periods in each coordinate (these are folded into the central period).
  // If pirange=0, the periodic domain for kx is [0,N1], ky [0,N2], kz [0,N3].
  // If pirange=1, the periodic domain is instead [-pi,pi] for each coord.
  // The SpreadParameters<FLT> struct must have been set up already by calling setup_kernel.
  // It is assumed that 2*opts.kernel_width < min(N1,N2,N3), so that the kernel
  // only ever wraps once when falls below 0 or off the top of a uniform grid
  // dimension.

  // Inputs:
  // N1,N2,N3 - grid sizes in x (fastest), y (medium), z (slowest) respectively.
  //           If N2==1, 1D spreading is done. If N3==1, 2D spreading.
  //     Otherwise, 3D.
  // M - number of NU pts.
  // kx, ky, kz - length-M real arrays of NU point coordinates (only kx read in
  //             1D, only kx and ky read in 2D).

  // These should lie in the box 0<=kx<=N1 etc (if pirange=0),
  //             or -pi<=kx<=pi (if pirange=1). However, points up to +-1 period
  //             outside this domain are also correctly folded back into this
  //             domain, but pts beyond this either raise an error (if check_bounds=1)
  //             or a crash (if check_bounds=0).
  // opts - spread/interp options struct, documented in ../include/SpreadParameters<FLT>.h

  // Inputs/Outputs:
  // data_uniform - output values on grid (dir=1) OR input grid data (dir=2)
  // data_nonuniform - input strengths of the sources (dir=1)
  //                   OR output values at targets (dir=2)
  // Returned value:
  // 0 indicates success; other values have meanings in ../docs/error.rst, with
  // following modifications:
  //   3 : one or more non-trivial box dimensions is less than 2.kernel_width.
  //   4 : nonuniform points outside [-Nm,2*Nm] or [-3pi,3pi] in at least one
  //       dimension m=1,2,3.
  //   5 : failed allocate sort indices

  // Magland Dec 2016. Barnett openmp version, many speedups 1/16/17-2/16/17
  // error codes 3/13/17. pirange 3/28/17. Rewritten 6/15/17. parallel sort 2/9/18
  // No separate subprob indices in t-1 2/11/18.
  // sort_threads (since for M<<N, multithread sort slower than single) 3/27/18
  // kereval, pad_kernel 4/24/18
  // Melody Shih split into 3 routines: check, sort, spread. Jun 2018, making
  // this routine just a caller to them. Name change, Barnett 7/27/18
  // Tidy, Barnett 5/20/20. Tidy doc, Barnett 10/22/20.
  Status spread_or_interp(DType* c, DType* f);

  // Spreads (or interpolates) a batch of batch_size strength vectors in cBatch
  // to (or from) the batch of fine working grids this->grid_data_, using the same set of
  // (index-sorted) NU points this->points_[0],Y,Z for each vector in the batch.
  // The direction (spread vs interpolate) is set by this->spread_params_.spread_direction.
  // Returns 0 (no error reporting for now).
  // Notes:
  // 1) cBatch is already assumed to have the correct offset, ie here we
  //    read from the start of cBatch (unlike Malleo). grid_data_ also has zero offset
  // 2) this routine is a batched version of spreadinterpSorted in spreadinterp.cpp
  // Barnett 5/19/20, based on Malleo 2019.
  // 3) the 3rd parameter is used when doing interp/spread only. When received,
  //    input/output data is read/written from/to this pointer instead of from/to
  //    the internal array this->fWBatch. Montalt 5/8/2021
  Status spread_or_interp_sorted_batch(
      int batch_size, DType* cBatch, DType* fBatch=nullptr);

  // Type 1: deconvolves (amplifies) from each interior fw array in this->grid_data_
  // into each output array fk in fkBatch.
  // Type 2: deconvolves from user-supplied input fk to 0-padded interior fw,
  // again looping over fk in fkBatch and fw in this->grid_data_.
  // The direction (spread vs interpolate) is set by this->spread_params_.spread_direction.
  // This is mostly a loop calling deconvolveshuffle?d for the needed rank batch_size
  // times.
  // Barnett 5/21/20, simplified from Malleo 2019 (eg t3 logic won't be in here)
  Status deconvolve_batch(int batch_size, DType* fkBatch);

 public:  // TODO(jmontalt): make private after refactoring FINUFFT.

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

  explicit Plan(OpKernelContext* context)
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
  std::unique_ptr<se::fft::Plan> fft_plan_;
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
#endif  // GOOGLE_CUDA

}  // namespace nufft
}  // namespace tensorflow

#endif  // TENSORFLOW_NUFFT_CC_KERNELS_NUFFT_PLAN_H_
