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

#include "tensorflow_nufft/cc/kernels/nufft_plan.h"

#include "tensorflow_nufft/cc/kernels/nufft_util.h"
#include "tensorflow_nufft/cc/kernels/omp_api.h"


namespace tensorflow {
namespace nufft {

namespace {

// Forward declarations. Defined below.
template<typename FloatType>
Status set_grid_size(int ms,
                     const Options& options,
                     SpreadParameters<FloatType> spread_params,
                     int* grid_size);

template<typename T>
Status reverse_vector(const gtl::InlinedVector<T, 4>& vec,
                      gtl::InlinedVector<T, 4>& rev);

template<typename FloatType>
Status setup_spreader(int rank, FloatType eps, double upsampling_factor,
                      int kerevalmeth, bool show_warnings,
                      SpreadParameters<FloatType> &spread_params);

template<typename FloatType>
Status setup_spreader_for_nufft(int rank, FloatType eps,
                                const Options& options,
                                SpreadParameters<FloatType> &spread_params);

}  // namespace

template<typename FloatType>
Plan<CPUDevice, FloatType>::Plan(
    OpKernelContext* context,
    TransformType type,
    int rank,
    gtl::InlinedVector<int, 4> num_modes,
    FftDirection fft_direction,
    int num_transforms,
    FloatType tol,
    const Options& options)
    : PlanBase<CPUDevice, FloatType>(context) {

  OP_REQUIRES(context,
              type != TransformType::TYPE_3,
              errors::Unimplemented("type-3 transforms are not implemented"));
  OP_REQUIRES(context, rank >= 1 && rank <= 3,
              errors::InvalidArgument("rank must be 1, 2 or 3"));
  OP_REQUIRES(context, num_transforms >= 1,
              errors::InvalidArgument("num_transforms must be >= 1"));
  OP_REQUIRES(context, rank == num_modes.size(),
              errors::InvalidArgument("num_modes must have size equal to rank"));

  // Store input values to plan.
  this->rank_ = rank;
  this->type_ = type;
  this->fft_direction_ = fft_direction;
  this->tol_ = tol;
  this->num_transforms_ = num_transforms;
  this->options_ = options;
  this->num_modes_ = num_modes;

  // Choose kernel evaluation method.
  if (this->options_.kernel_evaluation_method == KernelEvaluationMethod::AUTO) {
    this->options_.kernel_evaluation_method = KernelEvaluationMethod::HORNER;
  }

  // Choose overall number of threads.
  int num_threads = OMP_GET_MAX_THREADS();
  if (this->options_.num_threads > 0)
    num_threads = this->options_.num_threads; // user override
  this->options_.num_threads = num_threads;   // update options_ with actual number

  // Select batch size.
  if (this->options_.max_batch_size == 0) {
    this->num_batches_ = 1 + (num_transforms - 1) / num_threads;
    this->batch_size_ = 1 + (num_transforms - 1) / this->num_batches_;
  } else {
    this->batch_size_ = std::min(this->options_.max_batch_size, num_transforms);
    this->num_batches_ = 1 + (num_transforms - 1) / this->batch_size_;
  }

  // Choose default spreader threading configuration.
  if (this->options_.spread_threading == SpreadThreading::AUTO)
    this->options_.spread_threading = SpreadThreading::PARALLEL_SINGLE_THREADED;

  // Read in user Fourier mode array sizes.
  if (type != TransformType::TYPE_3) {
    // this->num_modes_[0] = num_modes[0];
    // this->num_modes_[1] = (rank > 1) ? num_modes[1] : 1;
    // this->num_modes_[2] = (rank > 2) ? num_modes[2] : 1;
    this->num_modes_total_ = this->num_modes_[0];
    if (rank > 1)
      this->num_modes_total_ *= this->num_modes_[1];
    if (rank > 2)
      this->num_modes_total_ *= this->num_modes_[2];
  }

  // Heuristic to choose default upsampling factor.
  if (this->options_.upsampling_factor == 0.0) {  // indicates auto-choose
    this->options_.upsampling_factor = 2.0;       // default, and need for tol small
    if (tol >= (FloatType)1E-9) {                   // the tol sigma=5/4 can reach
      if (type == TransformType::TYPE_3)
        this->options_.upsampling_factor = 1.25;  // faster b/c smaller RAM & FFT
      else if ((rank==1 && this->num_modes_total_>10000000) || (rank==2 && this->num_modes_total_>300000) || (rank==3 && this->num_modes_total_>3000000))  // type 1,2 heuristic cutoffs, double, typ tol, 12-core xeon
        this->options_.upsampling_factor = 1.25;
    }
    if (this->options_.verbosity > 1)
      printf("[%s] set auto upsampling_factor=%.2f\n",__func__,this->options_.upsampling_factor);
  }

  // Populate the spreader options.
  OP_REQUIRES_OK(context,
                 setup_spreader_for_nufft(
                    rank, tol, this->options_, this->spread_params_));

  // set others as defaults (or unallocated for arrays)...
  this->X = nullptr; this->Y = nullptr; this->Z = nullptr;
  this->phiHat1 = nullptr; this->phiHat2 = nullptr; this->phiHat3 = nullptr;
  this->sortIndices = nullptr;               // used in all three types
  
  // FFTW initialization must be done single-threaded.
  #pragma omp critical
  {
    static bool is_fftw_initialized = false;

    if (!is_fftw_initialized) {
      // Set up global FFTW state. Should be done only once.
      #ifdef _OPENMP
      // Initialize FFTW threads.
      fftw::init_threads<FloatType>();
      // Let FFTW use all threads.
      fftw::plan_with_nthreads<FloatType>(num_threads);
      #endif
      is_fftw_initialized = true;
    }
  }

  if (type == TransformType::TYPE_1)
    this->spread_params_.spread_direction = SpreadDirection::SPREAD;
  else // if (type == TransformType::TYPE_2)
    this->spread_params_.spread_direction = SpreadDirection::INTERP;
  
  // Determine fine grid sizes.
  this->grid_sizes_.resize(rank);
  OP_REQUIRES_OK(
      context, set_grid_size(
          this->num_modes_[0], this->options_, this->spread_params_, &this->grid_sizes_[0]));
  if (rank > 1) {
    OP_REQUIRES_OK(
        context, set_grid_size(
            this->num_modes_[1], this->options_, this->spread_params_, &this->grid_sizes_[1]));
  }
  if (rank > 2) {
    OP_REQUIRES_OK(
        context, set_grid_size(
            this->num_modes_[2], this->options_, this->spread_params_, &this->grid_sizes_[2]));
  }

  // Get Fourier coefficients of spreading kernel along each fine grid
  // dimension.
  this->phiHat1 = (FloatType*) malloc(sizeof(FloatType) * (this->grid_sizes_[0] / 2 + 1));
  kernel_fseries_1d(this->grid_sizes_[0], this->spread_params_, this->phiHat1);
  if (rank > 1) {
    this->phiHat2 = (FloatType*) malloc(sizeof(FloatType) * (this->grid_sizes_[1] / 2 + 1));
    kernel_fseries_1d(this->grid_sizes_[1], this->spread_params_, this->phiHat2);
  }
  if (rank > 2) {
    this->phiHat3 = (FloatType*) malloc(sizeof(FloatType) * (this->grid_sizes_[2] / 2 + 1));
    kernel_fseries_1d(this->grid_sizes_[2], this->spread_params_, this->phiHat3);
  }

  // Total number of points in the fine grid.
  this->num_grid_points_ = static_cast<int64_t>(this->grid_sizes_[0]);
  if (rank > 1)
    this->num_grid_points_ *= static_cast<int64_t>(this->grid_sizes_[1]);
  if (rank > 2)
    this->num_grid_points_ *= static_cast<int64_t>(this->grid_sizes_[2]);

  OP_REQUIRES(context, this->num_grid_points_ * this->batch_size_ <= kMaxArraySize,
              errors::Internal(
                  "size of internal fine grid is larger than maximum allowed: ",
                  this->num_grid_points_ * this->batch_size_, " > ",
                  kMaxArraySize));

  // Allocate the working fine grid through the op kernel context. We allocate a
  // flat array, since we'll only use this tensor through a raw pointer anyway.
  TensorShape fine_grid_shape({this->num_grid_points_ * this->batch_size_});
  OP_REQUIRES_OK(context,
                 context->allocate_temp(
                    DataTypeToEnum<DType>::value,
                    fine_grid_shape, &this->grid_tensor_));
  this->grid_data_ = reinterpret_cast<FftwType*>(
      this->grid_tensor_.flat<DType>().data());

  gtl::InlinedVector<int, 4> fft_dims;
  OP_REQUIRES_OK(context, reverse_vector(this->grid_sizes_, fft_dims));

  #pragma omp critical
  {
    this->fft_plan_ = fftw::plan_many_dft<FloatType>(
        /* int rank */ rank, /* const int *n */ fft_dims.data(),
        /* int howmany */ this->batch_size_,
        /* fftw_complex *in */ this->grid_data_,
        /* const int *inembed */ nullptr,
        /* int istride */ 1, /* int idist */ this->num_grid_points_,
        /* fftw_complex *out */ this->grid_data_,
        /* const int *onembed */ nullptr,
        /* int ostride */ 1, /* int odist */ this->num_grid_points_,
        /* int sign */ static_cast<int>(this->fft_direction_),
        /* unsigned flags */ this->options_.fftw_flags);
  }
}

template<typename FloatType>
Plan<CPUDevice, FloatType>::~Plan() {

  // Destroy the FFTW plan. This must be done single-threaded.
  #pragma omp critical
  {
    fftw::destroy_plan<FloatType>(this->fft_plan_);
  }
  
  // Wait until all threads are done using FFTW, then clean up the FFTW state,
  // which only needs to be done once.
  #ifdef _OPENMP
  #pragma omp barrier
  #pragma omp critical
  {
    static bool is_fftw_finalized = false;
    if (!is_fftw_finalized) {
      fftw::cleanup_threads<FloatType>();
      is_fftw_finalized = true;
    }
  }
  #endif

  free(this->sortIndices);
  free(this->phiHat1);
  free(this->phiHat2);
  free(this->phiHat3);
}

namespace {

// Set 1D size of upsampled array, grid_size, given options and requested number of
// Fourier modes.
template<typename FloatType>
Status set_grid_size(int ms,
                     const Options& options,
                     SpreadParameters<FloatType> spread_params,
                     int* grid_size) {
  // for spread/interp only, we do not apply oversampling (Montalt 6/8/2021).
  if (options.spread_only) {
    *grid_size = ms;
  } else {
    *grid_size = static_cast<int>(options.upsampling_factor * ms);
  }

  // This is required to avoid errors.
  if (*grid_size < 2 * spread_params.kernel_width)
    *grid_size = 2 * spread_params.kernel_width;

  // Check if array size is too big.
  if (*grid_size > kMaxArraySize) {
    return errors::Internal(
        "Upsampled dim size too big: ", *grid_size, " > ", kMaxArraySize);
  }

  // Find the next smooth integer.
  *grid_size = next_smooth_int(*grid_size);

  // For spread/interp only mode, make sure that the grid size is valid.
  if (options.spread_only && *grid_size != ms) {
    return errors::Internal(
        "Invalid grid size: ", ms, ". Value should be even, "
        "larger than the kernel (", 2 * spread_params.kernel_width, ") and have no prime "
        "factors larger than 5.");
  }

  return Status::OK();
}

// Reverses the input vector. Only supports up to rank 3 (this is not checked).
template<typename T>
Status reverse_vector(const gtl::InlinedVector<T, 4>& vec,
                      gtl::InlinedVector<T, 4>& rev) {

  if (vec.size() > 3)
    return errors::InvalidArgument("rank > 3 not supported");

  rev.resize(vec.size());

  if (vec.size() == 1) { 
    rev[0] = vec[0];
  }
  else if (vec.size() == 2) {
    rev[0] = vec[1];
    rev[1] = vec[0];
  }
  else if (vec.size() == 3) {
    rev[0] = vec[2];
    rev[1] = vec[1];
    rev[2] = vec[0];
  }

  return Status::OK();
}

template<typename FloatType>
Status setup_spreader(
    int rank, FloatType eps, double upsampling_factor,
    int kerevalmeth, bool show_warnings, SpreadParameters<FloatType> &spread_params)
/* Initializes spreader kernel parameters given desired NUFFT tol eps,
   upsampling factor (=sigma in paper, or R in Dutt-Rokhlin), ker eval meth
   (either 0:exp(sqrt()), 1: Horner ppval), and some debug-level flags.
   Also sets all default options in SpreadParameters<FloatType>. See SpreadParameters<FloatType>.h for spread_params.
   rank is spatial dimension (1,2, or 3).
   See finufft.cpp:finufft_plan() for where upsampling_factor is set.
   Must call this before any kernel evals done, otherwise segfault likely.
   Returns:
     0  : success
     WARN_EPS_TOO_SMALL : requested eps cannot be achieved, but proceed with
                          best possible eps
     otherwise : failure (see codes in finufft_definitions.h); spreading must not proceed
   Barnett 2017. debug, loosened eps logic 6/14/20.
*/
{
  if (upsampling_factor != 2.0 && upsampling_factor != 1.25) {
    if (kerevalmeth == 1) {
      return errors::Internal(
          "Horner kernel evaluation only supports standard "
          "upsampling factors of 2.0 or 1.25, but got ", upsampling_factor);
    }
    if (upsampling_factor <= 1.0) {
      return errors::Internal(
          "upsampling_factor must be > 1.0, but is ", upsampling_factor);
    }
  }
    
  // write out default SpreadParameters<FloatType>
  spread_params.pirange = 1;             // user also should always set this
  spread_params.check_bounds = false;
  spread_params.sort_points = SortPoints::AUTO;
  spread_params.pad_kernel = 0;              // affects only evaluate_kernel_vector
  spread_params.kerevalmeth = kerevalmeth;
  spread_params.upsampling_factor = upsampling_factor;
  spread_params.num_threads = 0;            // all avail
  spread_params.sort_threads = 0;        // 0:auto-choice
  // heuristic dir=1 chunking for nthr>>1, typical for intel i7 and skylake...
  spread_params.max_subproblem_size = (rank == 1) ? 10000 : 100000;
  spread_params.flags = 0;               // 0:no timing flags (>0 for experts only)
  spread_params.verbosity = 0;               // 0:no debug output
  // heuristic nthr above which switch OMP critical to atomic (add_wrapped...):
  spread_params.atomic_threshold = 10;   // R Blackwell's value

  int ns = 0;  // Set kernel width w (aka ns, kernel_width) then copy to spread_params...
  if (eps < kEpsilon<FloatType>) {
    eps = kEpsilon<FloatType>;
  }

  // Select kernel width.
  if (upsampling_factor == 2.0)           // standard sigma (see SISC paper)
    ns = std::ceil(-log10(eps / (FloatType)10.0));          // 1 digit per power of 10
  else                          // custom sigma
    ns = std::ceil(-log(eps) / (kPi<FloatType> * sqrt(1.0 - 1.0 / upsampling_factor)));  // formula, gam=1
  ns = std::max(2, ns);               // (we don't have ns=1 version yet)
  if (ns > kMaxKernelWidth) {         // clip to fit allocated arrays, Horner rules
    ns = kMaxKernelWidth;
  }
  spread_params.kernel_width = ns;

  // setup for reference kernel eval (via formula): select beta width param...
  // (even when kerevalmeth=1, this ker eval needed for FTs in onedim_*_kernel)
  spread_params.kernel_half_width = (FloatType)ns / 2;   // constants to help (see below routines)
  spread_params.kernel_c = 4.0 / (FloatType)(ns * ns);
  FloatType beta_over_ns = 2.30;         // gives decent betas for default sigma=2.0
  if (ns == 2) beta_over_ns = 2.20;  // some small-width tweaks...
  if (ns == 3) beta_over_ns = 2.26;
  if (ns == 4) beta_over_ns = 2.38;
  if (upsampling_factor != 2.0) {          // again, override beta for custom sigma
    FloatType gamma = 0.97;              // must match devel/gen_all_horner_C_code.m !
    beta_over_ns = gamma * kPi<FloatType>*(1.0 - 1.0 / (2 * upsampling_factor));  // formula based on cutoff
  }
  spread_params.kernel_beta = beta_over_ns * (FloatType)ns;    // set the kernel beta parameter

  // Calculate scaling factor for spread/interp only mode.
  if (spread_params.spread_only)
    spread_params.kernel_scale = calculate_scale_factor<FloatType>(rank, spread_params);

  return Status::OK();
}

template<typename FloatType>
Status setup_spreader_for_nufft(int rank, FloatType eps,
                                const Options& options,
                                SpreadParameters<FloatType> &spread_params)
// Set up the spreader parameters given eps, and pass across various nufft
// options. Return status of setup_spreader. Uses pass-by-ref. Barnett 10/30/17
{
  // This must be set before calling setup_spreader
  spread_params.spread_only = options.spread_only;

  TF_RETURN_IF_ERROR(setup_spreader(
      rank, eps, options.upsampling_factor,
      static_cast<int>(options.kernel_evaluation_method) - 1, // We subtract 1 temporarily, as spreader expects values of 0 or 1 instead of 1 and 2.
      options.show_warnings, spread_params));

  // override various spread spread_params from their defaults...
  spread_params.sort_points = options.sort_points;
  spread_params.spread_method = options.spread_method;
  spread_params.verbosity = options.verbosity;
  spread_params.pad_kernel = options.pad_kernel; // (only applies to kerevalmeth=0)
  spread_params.check_bounds = options.check_bounds;
  spread_params.num_threads = options.num_threads;
  if (options.num_threads_for_atomic_spread >= 0) // overrides
    spread_params.atomic_threshold = options.num_threads_for_atomic_spread;
  if (options.max_spread_subproblem_size > 0)        // overrides
    spread_params.max_subproblem_size = options.max_spread_subproblem_size;

  return Status::OK();
}

}  // namespace

// Explicit instatiations.
template class Plan<CPUDevice, float>;
template class Plan<CPUDevice, double>;

}  // namespace nufft
}  // namespace tensorflow
