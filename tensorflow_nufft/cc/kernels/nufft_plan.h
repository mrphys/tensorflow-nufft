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

#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuComplex.h"
#include "third_party/gpus/cuda/include/cufft.h"
#endif
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow_nufft/cc/kernels/fftw_api.h"
#include "tensorflow_nufft/cc/kernels/legendre_rule_fast.h"
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
constexpr static FloatType kTwoPi = FloatType(6.283185307179586);

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

// Parameters for the interpolation kernel.
template<typename FloatType>
struct KernelArgs {
  int w;  // width
  FloatType hw;  // half-width
  FloatType beta;  // constant beta
  FloatType c;  // constant c
  FloatType scale;  // scaling factor
};

template<typename FloatType>
struct SpreadParameters {
  // Number of threads used during sorting. If 0, a default value is selected.
  // Only relevant for CPU implementation.
  int sort_threads;
  // Number of threads above which spreading is performed using atomics.
  int atomic_threshold;
  // TODO(jmontalt): revise the following options.
  int pirange;            // 0: NU periodic domain is [0,N), 1: domain [-pi,pi)
  int kerevalmeth;        // 0: direct exp(sqrt()), or 1: Horner ppval, fastest
  int max_subproblem_size;  // # pts per t1 subprob; sets extra RAM per thread
  int flags;              // binary flags for timing only (may give wrong ans
                          // if changed from 0!). See spreadinterp.h
  int verbosity;          // 0: silent, 1: small text output, 2: verbose
};

namespace {
// Represents the Thrust execution policy type, which is currently specialized
// for CPU and GPU.
// Note: It might be possible to avoid the need for this struct by using
// the base type `thrust::execution_policy`, but it is not clear to me how to
// do this since `thrust::execution_policy` is itself a template which depends
// on the derived execution policy.
template<typename Device>
class ExecutionPolicy;

template<>
struct ExecutionPolicy<CPUDevice> {
  using Type = thrust::system::cpp::detail::par_t;
};

#if GOOGLE_CUDA
template<>
struct ExecutionPolicy<GPUDevice> {
  using Type = thrust::system::cuda::detail::par_t::stream_attachment_type;
};
#endif  // GOOGLE_CUDA
}  // namespace

template<typename Device, typename FloatType>
class PlanBase {
 public:
  // The main data type this plan operates with; either complex float or double.
  using DType = typename ComplexType<Device, FloatType>::Type;
  using ExecutionPolicyType = typename ExecutionPolicy<Device>::Type;

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
                            int* grid_dims,
                            FftDirection fft_direction,
                            int num_transforms,
                            FloatType tol,
                            const InternalOptions& options) = 0;

  // Sets the number and coordinates of the non-uniform points. Allocates arrays
  // arrays that depend on the number of points. Maybe sorts the non-uniform
  // points. Maybe rescales the non-uniform points to the range used by the
  // spreader. Determines the spreader parameters that depend on the non-uniform
  // points. Must be called after `initialize`.
  //
  // Note: the plan does not take ownership of pointers `points_x`, `points_y`,
  // `points_z`. However, the plan may change the values in `points`. The
  // caller must ensure that the memory is valid until the plan is destroyed.
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

 protected:
  // initialize(...)

  // Sets default values for unset options.
  // Sets: options_.upsampling_factor, options_.kernel_eval_algo.
  // Requires: tol_ must be set.
  // this->options_ is valid after calling this function.
  Status set_default_options();

  // Returns the default upsampling factor.
  virtual double default_upsampling_factor() const = 0;

  // Returns the default kernel evaluation algorithm.
  virtual KernelEvalAlgo default_kernel_eval_algo() const = 0;

  // Checks that the kernel evaluation algorithm is valid.
  virtual Status check_kernel_eval_algo(KernelEvalAlgo algo) const = 0;

  // Initializes the interpolator.
  Status initialize_interpolator();

  // Initializes the interpolation kernel.
  // Sets: kernel_args_.
  Status initialize_kernel();

  // Initializes the fine grid dimension sizes and allocates the array.
  // Sets: fine_dims_, fine_size_, fine_tensor_, fine_data_.
  // Requires: rank_, tol_, grid_dims_, grid_size_ and batch_size_.
  Status initialize_fine_grid();

  // Initializes the deconvolution/amplification weights.
  virtual Status initialize_weights() = 0;

  // Approximates exact Fourier series coeffs of cnufftspread's real symmetric
  // kernel, directly via q-node quadrature on Euler-Fourier formula, exploiting
  // narrowness of kernel. Uses phase winding for cheap eval on the regular freq
  // grid. Note that this is also the Fourier transform of the non-periodized
  // kernel. The FT definition is f(k) = int e^{-ikx} f(x) dx. The output has an
  // overall prefactor of 1/h, which is needed anyway for the correction, and
  // arises because the quadrature weights are scaled for grid units not x
  // units.
  void compute_weights(int dim_size, FloatType* weights) const;

  // Evaluates the exponential of semi-circle kernel at the specified point.
  // Kernel is related to an asymptotic approximation to the Kaiser-Bessel
  // kernel, itself an approximation to prolate spheroidal wavefunction (PSWF)
  // of order 0.
  FloatType evaluate_kernel(FloatType x) const;

  // Validates the input fine grid dimension, returning a potentially
  // modified valid value. This is specialized for CPU and GPU.
  virtual int validate_fine_grid_dimension(int idx, int dim) const = 0;

  // Initializes the FFT library and plan.
  virtual Status initialize_fft() = 0;

  // set_points(...)

  // Checks that the nonuniform points are within the supported bounds.
  Status check_points_within_range() const;

  // Folds and rescales nonuniform points to the canonical range.
  Status fold_and_rescale_points();

  // Performs bin-sorting if required for plan configuration.
  virtual Status binsort_if_needed() = 0;

  // general

  // Returns the lower/upper bound of the nonuniform point coordinates.
  FloatType points_lower_bound(int dim) const;
  FloatType points_upper_bound(int dim) const;

  // Retrieves the default Thrust execution policy.
  virtual const ExecutionPolicyType execution_policy() const = 0;

 protected:
  // The rank of the transform (number of dimensions). Must be 1, 2 or 3.
  int rank_;

  // The type of the transform. See enum above.
  TransformType type_;

  // Direction of the FFT. See enum above.
  FftDirection fft_direction_;

  // Direction of the spreading. See enum above.
  SpreadDirection spread_direction_;

  // Relative user tol.
  FloatType tol_;

  // Number of transforms to compute.
  int num_transforms_;

  // Number of transforms to be performed in a single batch.
  int batch_size_;

  // The grid's dimension sizes or number of modes along each dimension.
  int grid_dims_[3];

  // The total element count of the grid or .
  int grid_size_;

  // The fine (oversampled) grid's dimension sizes.
  // Unused dimensions are set to 1.
  int fine_dims_[3];

  // The total element count of the fine (oversampled) grid.
  int fine_size_;

  // Batch of fine grids for FFT. This is usually the
  // largest array allocated by NUFFT.
  Tensor fine_tensor_;

  // A convenience pointer to the fine grid array for FFT calls.
  DType* fine_data_;

  // The total number of points.
  int num_points_;

  // Pointers to the non-uniform point coordinates. Each of these points to an
  // array of length `num_points_`.
  // Notes:
  //  - In the GPU implementation, these are device pointers.
  //  - These pointers are not owned by the plan.
  //  - Unused pointers are set to nullptr.
  FloatType* points_[3];

  // Parameters of the interpolation kernel.
  KernelArgs<FloatType> kernel_args_;

  // Pointer to the op kernel context.
  OpKernelContext* context_;

  // Reference to the active device.
  const Device& device_;

  // Advanced NUFFT options.
  InternalOptions options_;
};

template<typename Device, typename FloatType>
class Plan;

template<typename FloatType>
class Plan<CPUDevice, FloatType> : public PlanBase<CPUDevice, FloatType> {
 public:
  using DType = typename ComplexType<CPUDevice, FloatType>::Type;
  using ExecutionPolicyType = typename ExecutionPolicy<CPUDevice>::Type;

  explicit Plan(OpKernelContext* context)
      : PlanBase<CPUDevice, FloatType>(context) { }

  ~Plan();

  Status initialize(TransformType type,
                    int rank,
                    int* grid_dims,
                    FftDirection fft_direction,
                    int num_transforms,
                    FloatType tol,
                    const InternalOptions& options) override;

  Status set_points(int num_points,
                    FloatType* points_x,
                    FloatType* points_y,
                    FloatType* points_z) override;

  Status execute(DType* c, DType* f) override;

  Status interp(DType* c, DType* f) override;

  Status spread(DType* c, DType* f) override;

 protected:
  // Returns the default upsampling factor.
  virtual double default_upsampling_factor() const override;

  // Returns the default kernel evaluation algorithm.
  virtual KernelEvalAlgo default_kernel_eval_algo() const override;

  // Checks that the kernel evaluation algorithm is valid.
  virtual Status check_kernel_eval_algo(KernelEvalAlgo algo) const override;

  // Validates the input fine grid dimension, returning a potentially
  // modified valid value.
  int validate_fine_grid_dimension(int idx, int dim) const override;

  // Initializes the deconvolution/amplification weights.
  Status initialize_weights() override;

  // Initializes the FFT library and plan.
  // Sets this->fft_plan_.
  Status initialize_fft() override;

  // Performs bin-sorting if required for plan configuration.
  Status binsort_if_needed() override;

  // Performs bin-sorting single-threaded.
  Status binsort_singlethread();

  // Performs bin-sorting multi-threaded.
  Status binsort_multithread();

  // Retrieves the default Thrust execution policy.
  const ExecutionPolicyType execution_policy() const override {
    // TODO: consider using a multi-threaded policy.
    return thrust::cpp::par;
  }

  // Precomputed non-uniform point permutation, used to speed up spread/interp.
  int64_t* binsort_indices_;

  // Whether bin-sorting is being used. This is automatically set by
  // `binsort_if_needed()`.
  bool do_binsort_;

 protected:

  // Magland Dec 2016. Barnett openmp version, many speedups 1/16/17-2/16/17
  // error codes 3/13/17. pirange 3/28/17. Rewritten 6/15/17. parallel sort 2/9/18
  // No separate subprob indices in t-1 2/11/18.
  // sort_threads (since for M<<N, multithread sort slower than single) 3/27/18
  // Melody Shih split into 3 routines: check, sort, spread. Jun 2018, making
  // this routine just a caller to them. Name change, Barnett 7/27/18
  // Tidy, Barnett 5/20/20. Tidy doc, Barnett 10/22/20.
  Status spread_or_interp(DType* c, DType* f);

  // Spreads (or interpolates) a batch of batch_size strength vectors in cBatch
  // to (or from) the batch of fine working grids this->fine_data_, using the same set of
  // (index-sorted) NU points this->points_[0],Y,Z for each vector in the batch.
  // The direction (spread vs interpolate) is set by this->spread_direction.
  // Returns 0 (no error reporting for now).
  // Notes:
  // 1) cBatch is already assumed to have the correct offset, ie here we
  //    read from the start of cBatch (unlike Malleo). fine_data_ also has zero offset
  // 2) this routine is a batched version of spreadinterpSorted in spreadinterp.cpp
  // Barnett 5/19/20, based on Malleo 2019.
  // 3) the 3rd parameter is used when doing interp/spread only. When received,
  //    input/output data is read/written from/to this pointer instead of from/to
  //    the internal array this->fWBatch. Montalt 5/8/2021
  Status spread_or_interp_sorted_batch(
      int batch_size, DType* cBatch, DType* fBatch=nullptr);

  // Logic to select the main spreading (dir=1) vs interpolation (dir=2) routine.
  // See spreadinterp() above for inputs arguments and definitions.
  // Return value should always be 0 (no error reporting).
  // Split out by Melody Shih, Jun 2018; renamed Barnett 5/20/20.
  Status spread_or_interp_sorted(
      FloatType *data_uniform, FloatType *data_nonuniform) const;

  // Spread NU pts in sorted order to a uniform grid. See spreadinterp() for
  // doc.
  Status spread_sorted(
      FloatType *data_uniform, FloatType *data_nonuniform) const;

  Status interp_sorted(
      FloatType *data_uniform, FloatType *data_nonuniform) const;

  // Type 1: deconvolves (amplifies) from each interior fw array in this->fine_data_
  // into each output array fk in fkBatch.
  // Type 2: deconvolves from user-supplied input fk to 0-padded interior fw,
  // again looping over fk in fkBatch and fw in this->fine_data_.
  // The direction (spread vs interpolate) is set by this->spread_direction.
  // This is mostly a loop calling deconvolveshuffle?d for the needed rank batch_size
  // times.
  // Barnett 5/21/20, simplified from Malleo 2019 (eg t3 logic won't be in here)
  Status deconvolve_batch(int batch_size, DType* fkBatch);

  // 1D, 2D and 3D deconvolution / amplification.
  // These functions also shift frequencies according to the configured mode
  // order.
  void deconvolve_1d(
      DType* fk, DType* fw, FloatType prefactor = FloatType(1.0));
  void deconvolve_2d(
      DType* fk, DType* fw, FloatType prefactor = FloatType(1.0));
  void deconvolve_3d(
      DType* fk, DType* fw, FloatType prefactor = FloatType(1.0));

 public:  // TODO(jmontalt): make private after refactoring FINUFFT.

  // Number of batches in one execution (includes all the transforms in
  // num_transforms_).
  int num_batches_;
  // The FFTW plan for FFTs.
  typename fftw::PlanType<FloatType>::Type fft_plan_;

  // Tensors in host memory. Used for deconvolution. Empty in spread/interp
  // mode. Only the first `rank` tensors are allocated.
  Tensor weights_tensor_[3];
  // Convenience raw pointers to above tensors. Only the first `rank` pointers
  // are valid.
  FloatType* weights_data_[3];
};

#if GOOGLE_CUDA
template<typename FloatType>
class Plan<GPUDevice, FloatType> : public PlanBase<GPUDevice, FloatType> {
 public:
  // The main data type this plan operates with; either complex float or double.
  using DType = typename ComplexType<GPUDevice, FloatType>::Type;
  using ExecutionPolicyType = typename ExecutionPolicy<GPUDevice>::Type;

  explicit Plan(OpKernelContext* context)
      : PlanBase<GPUDevice, FloatType>(context) { }

  ~Plan();

  Status initialize(TransformType type,
                    int rank,
                    int* grid_dims,
                    FftDirection fft_direction,
                    int num_transforms,
                    FloatType tol,
                    const InternalOptions& options) override;

  Status set_points(int num_points,
                    FloatType* points_x,
                    FloatType* points_y,
                    FloatType* points_z) override;

  Status execute(DType* d_c, DType* d_fk) override;

  Status interp(DType* d_c, DType* d_fk) override;

  Status spread(DType* d_c, DType* d_fk) override;

 protected:
  static int64_t CufftScratchSize;

  // Returns the default upsampling factor.
  virtual double default_upsampling_factor() const override;

  // Returns the default kernel evaluation algorithm.
  virtual KernelEvalAlgo default_kernel_eval_algo() const override;

  // Checks that the kernel evaluation algorithm is valid.
  virtual Status check_kernel_eval_algo(KernelEvalAlgo algo) const override;

  // Validates the input fine grid dimension, returning a potentially
  // modified valid value.
  int validate_fine_grid_dimension(int idx, int dim) const override;

  // Initializes the deconvolution/amplification weights.
  Status initialize_weights() override;

  // Initializes the FFT library and plan.
  Status initialize_fft() override;

  // Performs bin-sorting if required for plan configuration.
  Status binsort_if_needed() override;

  // Initializes subproblems for subproblem-based interpolation/spreading.
  Status initialize_subproblems();

  // Retrieves the default Thrust execution policy.
  const ExecutionPolicyType execution_policy() const override {
    // TODO: consider using par_nosync once Thrust gets upgraded to 1.16.
    return thrust::cuda::par.on(this->device_.stream());
  }

 private:
  Status setup_spreader();
  Status spread_batch(int batch_size);
  Status interp_batch(int batch_size);
  Status spread_batch_nupts_driven(int batch_size);
  Status spread_batch_subproblem(int batch_size);
  Status interp_batch_nupts_driven(int batch_size);
  Status interp_batch_subproblem(int batch_size);
  // Deconvolve and/or amplify a batch of data.
  Status deconvolve_batch(int batch_size);
  // Tensors in device memory. Used for deconvolution. Empty in spread/interp
  // mode. Only the first `rank` tensors are allocated.
  Tensor weights_tensor_[3];
  // Convenience raw pointers to above tensors. These are device pointers. Only
  // the first `rank` pointers are valid.
  FloatType* weights_data_[3];
  // The cuFFT plan.
  std::unique_ptr<se::fft::Plan> fft_plan_;
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


namespace {
// Returns an even integer not less than n, with prime factors no larger than 5
// (ie, "smooth").
// If optional arg b is specified, the returned number must also be a multiple
// of b (b must be a number whose prime factors are no larger than 5).
template<typename IntType>
IntType next_smooth_integer(IntType n, IntType b = 1) {
  // If n smaller than two, return 2.
  if (n <= 2) return 2;
  // If n is odd, make even by adding 1.
  if (n % 2 == 1) n += 1;
  // Initialize loop. At each iteration, we add 2 to p, and then check if p is
  // smooth, by removing from p all factors of 2, 3 and 5, and storing the
  // result in d. At the end of the loop, d = 1 iff p is smooth.
  IntType p = n - 2;  // Subtract 2 to cancel out +2 in the loop.
  IntType d = 2;  // A dummy initialization value (must be > 1).
  while ((d > 1) || (p % b != 0)) {
    // Add 2 to stay even (odd numbers are never smooth).
    p += 2;
    // Remove all factors of 2, 3 and 5 from p, saving the result in d.
    d = p;
    while (d % 2 == 0) d /= 2;
    while (d % 3 == 0) d /= 3;
    while (d % 5 == 0) d /= 5;
  }
  return p;
}


// Functors.
// Note: We use functors instead of more convenient __host__ __device__
// lambda expressions because the latter may have reduced performance on the
// host due to the compiler's inability to inline them.
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#host-device-lambda-notes

// Checks if input is within the specified range.
template<typename FloatType>
struct IsWithinRange : public thrust::unary_function<FloatType, bool> {
  IsWithinRange(const FloatType& lower, const FloatType& upper)
    : lower_(lower), upper_(upper) { }

  __host__ __device__
  bool operator()(const FloatType& x) const {
    return (x > lower_) && (x < upper_);
  }

  FloatType lower_;
  FloatType upper_;
};


// Folds and rescales the input point to the canonical range `[0, n]`.
// There are specializations depending on the input points range.
template<typename FloatType, PointsRange>
struct FoldAndRescale : public thrust::unary_function<FloatType, FloatType> {

};


template<typename FloatType>
struct FoldAndRescale<FloatType, PointsRange::STRICT>
  : public thrust::unary_function<FloatType, FloatType> {
  FoldAndRescale(int n) : n_(n) { }

  __host__ __device__
  FloatType operator()(const FloatType& x) const {
    return (x + kPi<FloatType>) *
        kOneOverTwoPi<FloatType> * static_cast<FloatType>(n_);
  }

  int n_;
};


template<typename FloatType>
struct FoldAndRescale<FloatType, PointsRange::EXTENDED>
  : public thrust::unary_function<FloatType, FloatType> {
  FoldAndRescale(int n) : n_(n) { }

  __host__ __device__
  FloatType operator()(const FloatType& x) const {
    FloatType s;
    if (x > kPi<FloatType>) {
      s = x - kPi<FloatType>;
    } else if (x < -kPi<FloatType>) {
      s = x + FloatType(3.0) * kPi<FloatType>;
    } else {
      s = x + kPi<FloatType>;
    }
    return s * kOneOverTwoPi<FloatType> * static_cast<FloatType>(n_);
  }

  int n_;
};


template<typename FloatType>
struct FoldAndRescale<FloatType, PointsRange::INFINITE>
  : public thrust::unary_function<FloatType, FloatType> {
  FoldAndRescale(int n) : n_(n) { }

  __host__ __device__
  FloatType operator()(const FloatType& x) const {
    FloatType s = std::fmod(x + kPi<FloatType>, kTwoPi<FloatType>);
    if (s < FloatType(0.0)) {
      s += kTwoPi<FloatType>;
    }
    return s * kOneOverTwoPi<FloatType> * static_cast<FloatType>(n_);
  }

  int n_;
};

}  // namespace


template<typename Device, typename FloatType>
Status PlanBase<Device, FloatType>::set_default_options() {
  // Upsampling factor.
  double upsampling_factor = this->options_.upsampling_factor;
  if (upsampling_factor == 0.0) {
    // Use default upsampling factor.
    upsampling_factor = this->default_upsampling_factor();
  } else {
    // User-specified value. Do input checking.
    if (upsampling_factor <= 1.0) {
      return errors::InvalidArgument(
          "upsampling_factor must be > 1.0, but got: ", upsampling_factor);
    }
  }
  this->options_.upsampling_factor = upsampling_factor;

  // Kernel evaluation algorithm.
  KernelEvalAlgo kernel_eval_algo = this->options_.kernel_eval_algo;
  if (kernel_eval_algo == KernelEvalAlgo::AUTO) {
    kernel_eval_algo = this->default_kernel_eval_algo();
  } else {
    TF_RETURN_IF_ERROR(this->check_kernel_eval_algo(kernel_eval_algo));
  }
  this->options_.kernel_eval_algo = kernel_eval_algo;

  return OkStatus();
}


template<typename Device, typename FloatType>
Status PlanBase<Device, FloatType>::initialize_interpolator() {
  TF_RETURN_IF_ERROR(this->initialize_kernel());

  switch (this->type_) {
    case TransformType::TYPE_1: {  // spreading, NU -> U
      this->spread_direction_ = SpreadDirection::SPREAD;
      break;
    }
    case TransformType::TYPE_2: {  // interpolation, U -> NU
      this->spread_direction_ = SpreadDirection::INTERP;
      break;
    }
    default: {
      LOG(FATAL) << "Invalid transform type.";
    }
  }

  return OkStatus();
};


template<typename Device, typename FloatType>
Status PlanBase<Device, FloatType>::initialize_kernel() {
  // Kernel width.
  int width = 0;
  if (this->options_.upsampling_factor == 2.0) {
    // Special case for sigma == 2.0.
    width = std::ceil(-log10(this->tol_ / FloatType(10.0)));
  } else {
    // General case.
    width = std::ceil(
        -log(this->tol_) /
        (kPi<FloatType> * std::sqrt(
            1.0 - 1.0 / this->options_.upsampling_factor)));
  }
  // Kernel width must be at least 2 and no larger than kMaxKernelWidth.
  width = std::max(width, 2);
  width = std::min(width, kMaxKernelWidth);

  // beta divided by width.
  FloatType beta_over_width;
  if (this->options_.upsampling_factor == 2.0) {
    // Heuristic precomputed values for standard upsampling factors, with some
    // adjustments for small widths.
    if (width == 2) {
      beta_over_width = 2.20;
    } else if (width == 3) {
      beta_over_width = 2.26;
    } else if (width == 4) {
      beta_over_width = 2.38;
    } else {
      // Good generic value.
      beta_over_width = 2.30;
    }
  } else {
    FloatType gamma = 0.97;
    beta_over_width = gamma * kPi<FloatType> *
                      (1.0 - 1.0 / (2 * this->options_.upsampling_factor));
  }

  this->kernel_args_.w = width;
  this->kernel_args_.hw = static_cast<FloatType>(width) / FloatType(2.0);
  this->kernel_args_.c = 4.0 / static_cast<FloatType>(width * width);
  this->kernel_args_.beta = static_cast<FloatType>(width) * beta_over_width;

  return OkStatus();
};


template<typename Device, typename FloatType>
Status PlanBase<Device, FloatType>::initialize_fine_grid() {
  // Initialize fine grid dimensions to 1.
  for (int d = 0; d < this->rank_; d++) {
    this->fine_dims_[d] = 1;
  }
  this->fine_size_ = 1;

  // Determine the fine grid dimensions.
  for (int d = 0; d < this->rank_; d++) {
    if (this->options_.spread_only) {
      // Spread-only operation: no oversampling.
      this->fine_dims_[d] = this->grid_dims_[d];
    } else {
      // Apply oversampling.
      this->fine_dims_[d] = static_cast<int>(
          this->grid_dims_[d] * this->options_.upsampling_factor);
    }

    // Make sure fine grid is at least as large as the kernel.
    if (this->fine_dims_[d] < 2 * this->kernel_args_.w) {
      this->fine_dims_[d] = 2 * this->kernel_args_.w;
    }

    // Find the next smooth integer.
    this->fine_dims_[d] = this->validate_fine_grid_dimension(
        d, this->fine_dims_[d]);

    // For spread-only operation, make sure that the grid size is valid.
    if (this->options_.spread_only &&
        this->fine_dims_[d] != this->grid_dims_[d]) {
      return errors::InvalidArgument(
          "Invalid grid dimension size: ", this->grid_dims_[d],
          ". Grid dimension must be even, larger than the kernel (",
          2 * this->kernel_args_.w,
          ") and have no prime factors larger than 5.");
    }

    // Update the total grid size.
    this->fine_size_ *= this->fine_dims_[d];
  }

  // Check that the total grid size is not too big.
  if (this->fine_size_ * this->batch_size_ > kMaxArraySize) {
    return errors::InvalidArgument(
        "Fine grid is too big: size ", this->fine_size_ * this->batch_size_,
        " > ", kMaxArraySize);
  }

  // Allocate the working fine grid using the op kernel context.
  // We allocate a flat array, since we'll only use this tensor through
  // a raw pointer anyway.
  // This array is only needed if we're not doing a spread-only operation.
  if (!this->options_.spread_only) {
    TensorShape fine_shape({this->fine_size_ * this->batch_size_});
    TF_RETURN_IF_ERROR(this->context_->allocate_temp(
        DataTypeToEnum<std::complex<FloatType>>::value,
        fine_shape,
        &this->fine_tensor_));
    this->fine_data_ = reinterpret_cast<DType*>(
        this->fine_tensor_.flat<std::complex<FloatType>>().data());
  }

  return OkStatus();
}


template<typename Device, typename FloatType>
void PlanBase<Device, FloatType>::compute_weights(
    int dim_size, FloatType* weights) const {
  // Number of quadrature nodes in z (from 0 to J/2, reflections will be added).
  int q = static_cast<int>(2 + 3.0 * this->kernel_args_.hw);
  FloatType f[kMaxQuadNodes];
  double z[2 * kMaxQuadNodes];
  double w[2 * kMaxQuadNodes];

  // Only half the nodes used, eg on (0, 1).
  legendre_compute_glr(2 * q, z, w);

  // Set up nodes z[n] and values f[n].
  std::complex<FloatType> a[kMaxQuadNodes];
  for (int n = 0; n < q; ++n) {
    z[n] *= this->kernel_args_.hw;
    f[n] = this->kernel_args_.hw * static_cast<FloatType>(w[n]) *
            this->evaluate_kernel(static_cast<FloatType>(z[n]));
    a[n] = exp(
        2 * kPi<FloatType> * kImaginaryUnit<FloatType> *
        static_cast<FloatType>(dim_size / 2 - z[n]) /
        static_cast<FloatType>(dim_size));  // phase winding rates
  }
  int nout = dim_size / 2 + 1;                   // how many values we're writing to
  int nt = std::min(nout, this->options_.num_threads);
  std::vector<int> brk(nt + 1);        // start indices for each thread
  for (int t = 0; t <= nt; ++t)             // split nout mode indices btw threads
    brk[t] = (int)(0.5 + nout * t / (double)nt);

  #pragma omp parallel num_threads(nt)
  {                                     // each thread gets own chunk to do
    int t = OMP_GET_THREAD_NUM();
    std::complex<FloatType> aj[kMaxQuadNodes];    // phase rotator for this thread

    for (int n = 0; n < q; ++n)
      aj[n] = pow(a[n], (FloatType)brk[t]);    // init phase factors for chunk

    for (int j = brk[t]; j < brk[t + 1]; ++j) {          // loop along output array
      FloatType x = 0.0;                      // accumulator for answer at this j
      for (int n = 0; n < q; ++n) {
        x += f[n] * 2 * real(aj[n]);      // include the negative freq
        aj[n] *= a[n];                  // wind the phases
      }
      weights[j] = x;
    }
  }
}


template<typename Device, typename FloatType>
FloatType PlanBase<Device, FloatType>::evaluate_kernel(FloatType x) {
  if (abs(x) >= this->kernel_args_.hw)
    return 0.0;
  return exp(this->kernel_args_.beta *
             sqrt(1.0 - this->kernel_args_.c * x * x));
}


template<typename Device, typename FloatType>
Status PlanBase<Device, FloatType>::check_points_within_range() const {
  if (this->options_.points_range() == PointsRange::INFINITE) {
    // No need to check in this case, as all values are valid.
    return OkStatus();
  }

  // For each dimension.
  FloatType lower_bound, upper_bound;
  for (int d = 0; d < this->rank_; d++) {
    // Determine appropriate bounds depending on configuration.
    lower_bound = this->points_lower_bound(d);
    upper_bound = this->points_upper_bound(d);

    bool all_points_within_range = thrust::transform_reduce(
        this->execution_policy(),
        this->points_[d],
        this->points_[d] + this->num_points_,
        IsWithinRange<FloatType>(lower_bound, upper_bound),
        true,
        thrust::logical_and<bool>());

    if (!all_points_within_range) {
      return errors::InvalidArgument(
          "Found points outside expected range for dimension ", d,
          ". Valid range is [", lower_bound, ", ", upper_bound, "]. "
          "Check your points and/or set a less restrictive value for "
          "options.points_range.");
    }
  }

  return OkStatus();
}


template<typename Device, typename FloatType>
Status PlanBase<Device, FloatType>::fold_and_rescale_points() {
  if (this->options_.points_unit != PointsUnit::RADIANS_PER_SAMPLE) {
    return errors::Unimplemented(
        "fold_and_rescale_points is only implemented for ",
        "points_unit == RADIANS_PER_SAMPLE.");
  }

  switch (this->options_.points_range()) {
    case PointsRange::STRICT:
      for (int d = 0; d < this->rank_; d++) {
        thrust::transform(
            this->execution_policy(),
            this->points_[d],
            this->points_[d] + this->num_points_,
            this->points_[d],
            FoldAndRescale<FloatType, PointsRange::STRICT>(
                this->fine_dims_[d]));
      }
      break;
    case PointsRange::EXTENDED:
      for (int d = 0; d < this->rank_; d++) {
        thrust::transform(
            this->execution_policy(),
            this->points_[d],
            this->points_[d] + this->num_points_,
            this->points_[d],
            FoldAndRescale<FloatType, PointsRange::EXTENDED>(
                this->fine_dims_[d]));
      }
      break;
    case PointsRange::INFINITE:
      for (int d = 0; d < this->rank_; d++) {
        thrust::transform(
            this->execution_policy(),
            this->points_[d],
            this->points_[d] + this->num_points_,
            this->points_[d],
            FoldAndRescale<FloatType, PointsRange::INFINITE>(
                this->fine_dims_[d]));
      }
      break;
    default:
      LOG(FATAL) << "invalid points range";
  }

  return OkStatus();
}


template<typename Device, typename FloatType>
FloatType PlanBase<Device, FloatType>::points_lower_bound(int dim) const {
  return -this->points_upper_bound(dim);
}


template<typename Device, typename FloatType>
FloatType PlanBase<Device, FloatType>::points_upper_bound(int dim) const {
  FloatType upper_bound;
  switch (this->options_.points_unit) {
    case PointsUnit::CYCLES: {
      upper_bound = static_cast<FloatType>((this->grid_dims_[dim] + 1) / 2);
      break;
    }
    case PointsUnit::CYCLES_PER_SAMPLE: {
      upper_bound = FloatType(0.5);
      break;
    }
    case PointsUnit::RADIANS_PER_SAMPLE: {
      upper_bound = kPi<FloatType>;
      break;
    }
    default: {
      LOG(FATAL) << "invalid points unit";
    }
  }

  switch (this->options_.points_range()) {
    case PointsRange::STRICT: {
      break;
    }
    case PointsRange::EXTENDED: {
      upper_bound *= FloatType(3.0);
      break;
    }
    case PointsRange::INFINITE: {
      upper_bound = std::numeric_limits<FloatType>::infinity();
      break;
    }
    default: {
      LOG(FATAL) << "invalid points range";
    }
  }

  return upper_bound;
}


}  // namespace nufft
}  // namespace tensorflow

#endif  // TENSORFLOW_NUFFT_CC_KERNELS_NUFFT_PLAN_H_
