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

#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include "tensorflow_nufft/cc/kernels/fftw_api.h"
#include "tensorflow_nufft/cc/kernels/nufft_plan.h"
#include "tensorflow_nufft/cc/kernels/nufft_util.h"
#include "tensorflow_nufft/cc/kernels/omp_api.h"


// Largest possible kernel spread width per dimension, in fine grid points.
#define MAX_KERNEL_WIDTH 16


namespace tensorflow {
namespace nufft {

// This macro was used in the original FINUFFT code. Here it's just an
// identity mapping as its result is precomputed, but we're keeping it as a
// placeholder in case we want to compute its result it on the fly in the
// future.
#define FOLD_AND_RESCALE(x, N, p) (x)

namespace {

template<typename FloatType>
Status setup_spreader(int rank,
                      int kerevalmeth, bool show_warnings,
                      const InternalOptions& options,
                      SpreadParameters<FloatType> &spread_params);

template<typename FloatType>
Status setup_spreader_for_nufft(int rank,
                                const InternalOptions& options,
                                SpreadParameters<FloatType> &spread_params);

static int get_transform_rank(int64_t n1, int64_t n2, int64_t n3);

template<typename FloatType>
Status check_spread_inputs(int64_t n1, int64_t n2, int64_t n3,
                           int64_t num_points, FloatType *kx, FloatType *ky,
                           FloatType *kz, SpreadParameters<FloatType> opts);

template<typename FloatType>
int spreadinterpSorted(int* sort_indices,int64_t N1, int64_t N2, int64_t N3,
		             FloatType *data_uniform,int64_t M, FloatType *kx, FloatType *ky, FloatType *kz,
		             FloatType *data_nonuniform, tensorflow::nufft::SpreadParameters<FloatType> opts, int did_sort);

template<typename FloatType>
int interpSorted(int* sort_indices,int64_t N1, int64_t N2, int64_t N3,
		      FloatType *data_uniform,int64_t M, FloatType *kx, FloatType *ky, FloatType *kz,
		      FloatType *data_nonuniform, SpreadParameters<FloatType> opts, int did_sort);

template<typename FloatType>
int spreadSorted(int* sort_indices,int64_t N1, int64_t N2, int64_t N3,
		      FloatType *data_uniform,int64_t M, FloatType *kx, FloatType *ky, FloatType *kz,
		      FloatType *data_nonuniform, SpreadParameters<FloatType> opts, int did_sort);

template<typename FloatType>
static inline void set_kernel_args(FloatType *args, FloatType x, const SpreadParameters<FloatType>& opts);

template<typename FloatType>
static inline void evaluate_kernel_vector(FloatType *ker, FloatType *args, const SpreadParameters<FloatType>& opts, const int N);

template<typename FloatType>
static inline void eval_kernel_vec_Horner(FloatType *ker, const FloatType z, const int w, const SpreadParameters<FloatType> &opts);

template<typename FloatType>
void interp_line(FloatType *out,FloatType *du, FloatType *ker,int64_t i1,int64_t N1,int ns);

template<typename FloatType>
void interp_square(FloatType *out,FloatType *du, FloatType *ker1, FloatType *ker2, int64_t i1,int64_t i2,int64_t N1,int64_t N2,int ns);

template<typename FloatType>
void interp_cube(FloatType *out,FloatType *du, FloatType *ker1, FloatType *ker2, FloatType *ker3,
		 int64_t i1,int64_t i2,int64_t i3,int64_t N1,int64_t N2,int64_t N3,int ns);

template<typename FloatType>
void spread_subproblem_1d(int64_t off1, int64_t size1,FloatType *du0,int64_t M0,FloatType *kx0,
                          FloatType *dd0,const SpreadParameters<FloatType>& opts);

template<typename FloatType>
void spread_subproblem_2d(int64_t off1, int64_t off2, int64_t size1,int64_t size2,
                          FloatType *du0,int64_t M0,
			  FloatType *kx0,FloatType *ky0,FloatType *dd0,const SpreadParameters<FloatType>& opts);

template<typename FloatType>
void spread_subproblem_3d(int64_t off1,int64_t off2, int64_t off3, int64_t size1,
                          int64_t size2,int64_t size3,FloatType *du0,int64_t M0,
			  FloatType *kx0,FloatType *ky0,FloatType *kz0,FloatType *dd0,
			  const SpreadParameters<FloatType>& opts);

template<typename FloatType>
void add_wrapped_subgrid(int64_t offset1,int64_t offset2,int64_t offset3,
			 int64_t size1,int64_t size2,int64_t size3,int64_t N1,
			 int64_t N2,int64_t N3,FloatType *data_uniform, FloatType *du0);

template<typename FloatType>
void add_wrapped_subgrid_thread_safe(int64_t offset1,int64_t offset2,int64_t offset3,
                                     int64_t size1,int64_t size2,int64_t size3,int64_t N1,
                                     int64_t N2,int64_t N3,FloatType *data_uniform, FloatType *du0);

template<typename FloatType>
void get_subgrid(int64_t &offset1,int64_t &offset2,int64_t &offset3,int64_t &size1,
		 int64_t &size2,int64_t &size3,int64_t M0,FloatType* kx0,FloatType* ky0,
		 FloatType* kz0,int ns, int ndims);

}  // namespace

template<typename FloatType>
Plan<CPUDevice, FloatType>::~Plan() {

  if (!this->options_.spread_only) {
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
  }
}

template<typename FloatType>
Status Plan<CPUDevice, FloatType>::initialize(
    TransformType type,
    int rank,
    int* grid_dims,
    FftDirection fft_direction,
    int num_transforms,
    FloatType tol,
    const InternalOptions& options) {
  if (type == TransformType::TYPE_3) {
    return errors::Unimplemented("type-3 transforms are not implemented");
  }
  if (rank < 1 || rank > 3) {
    return errors::Unimplemented("rank ", rank, " is not implemented");
  }
  if (num_transforms < 1) {
    return errors::InvalidArgument("num_transforms must be >= 1");
  }

  // Store input values to plan.
  this->rank_ = rank;
  this->type_ = type;
  this->fft_direction_ = fft_direction;
  this->tol_ = std::max(tol, kEpsilon<FloatType>);
  this->num_transforms_ = num_transforms;
  this->options_ = options;

  this->grid_dims_[0] = grid_dims[0];
  this->grid_dims_[1] = (this->rank_ > 1) ? grid_dims[1] : 1;
  this->grid_dims_[2] = (this->rank_ > 2) ? grid_dims[2] : 1;
  this->grid_size_ = this->grid_dims_[0] * this->grid_dims_[1] * this->grid_dims_[2];

  // Choose kernel evaluation method.
  if (this->options_.kernel_evaluation_method == KernelEvaluationMethod::AUTO) {
    this->options_.kernel_evaluation_method = KernelEvaluationMethod::HORNER;
  }

  // Choose overall number of threads.
  int num_threads = OMP_GET_MAX_THREADS();
  if (this->options_.num_threads > 0)
    num_threads = this->options_.num_threads;
  this->options_.num_threads = num_threads;
  // Set number of threads to be used in subsequent parallel regions.
  OMP_SET_NUM_THREADS(this->options_.num_threads);

  // Select batch size.
  if (this->options_.max_batch_size() == 0) {
    this->num_batches_ = 1 + (num_transforms - 1) / num_threads;
    this->batch_size_ = 1 + (num_transforms - 1) / this->num_batches_;
  } else {
    this->batch_size_ = std::min(
        this->options_.max_batch_size(), num_transforms);
    this->num_batches_ = 1 + (num_transforms - 1) / this->batch_size_;
  }

  TF_RETURN_IF_ERROR(this->set_default_options());

  TF_RETURN_IF_ERROR(this->initialize_interpolator());

  TF_RETURN_IF_ERROR(this->initialize_fine_grid());

  // Choose default spreader threading configuration.
  // TODO: move to set_default_options.
  if (this->options_.spread_threading == SpreadThreading::AUTO)
    this->options_.spread_threading = SpreadThreading::PARALLEL_SINGLE_THREADED;

  // Populate the spreader options.
  TF_RETURN_IF_ERROR(setup_spreader_for_nufft(
      rank, this->options_, this->spread_params_));

  // Initialize pointers to null.
  for (int i = 0; i < 3; i++) {
    this->points_[i] = nullptr;
    this->fseries_data_[i] = nullptr;
  }

  // TODO(jmontalt): Remove once spread_params_ has been factored away.
  this->spread_params_.spread_direction = this->spread_direction_;

  // Get Fourier coefficients of spreading kernel along each fine grid
  // dimension.
  for (int i = 0; i < this->rank_; i++) {
    // Number of Fourier coefficients.
    int num_coeffs = this->fine_dims_[i] / 2 + 1;
    // Allocate memory and calculate the Fourier series.
    TF_RETURN_IF_ERROR(this->context_->allocate_temp(
        DataTypeToEnum<FloatType>::value, TensorShape({num_coeffs}),
        &this->fseries_tensor_[i]));
    this->fseries_data_[i] = reinterpret_cast<FloatType*>(
        this->fseries_tensor_[i].flat<FloatType>().data());
    kernel_fseries_1d(this->fine_dims_[i], this->spread_params_,
                      this->fseries_data_[i]);
  }

  if (!this->options_.spread_only) {
    TF_RETURN_IF_ERROR(this->initialize_fft());
  }

  return OkStatus();
}

template<typename FloatType>
Status Plan<CPUDevice, FloatType>::set_points(
    int num_points,
    FloatType* points_x,
    FloatType* points_y,
    FloatType* points_z) {
  // The user only now chooses how many NU (x,y,z) points.
  this->num_points_ = num_points;
  this->points_[0] = points_x;
  this->points_[1] = this->rank_ > 1 ? points_y : nullptr;
  this->points_[2] = this->rank_ > 2 ? points_z : nullptr;

  int64_t grid_size_0 = this->fine_dims_[0];
  int64_t grid_size_1 = 1;
  if (this->rank_ > 1) grid_size_1 = this->fine_dims_[1];
  int64_t grid_size_2 = 1;
  if (this->rank_ > 2) grid_size_2 = this->fine_dims_[2];

  TF_RETURN_IF_ERROR(check_spread_inputs(
      grid_size_0, grid_size_1, grid_size_2,
      this->num_points_, points_x, points_y, points_z,
      this->spread_params_));

  // Check that points are within bounds.
  if (this->options_.debugging().check_points_range()) {
    TF_RETURN_IF_ERROR(this->check_points_within_range());
  }

  // Fold and rescale points.
  TF_RETURN_IF_ERROR(this->fold_and_rescale_points());

  // Bin-sorting for efficient RAM access.
  TF_RETURN_IF_ERROR(this->binsort_if_needed());

  return OkStatus();
}

/* See ../docs/cguru.doc for current documentation.

   For given (stack of) weights cj or coefficients fk, performs NUFFTs with
   existing (sorted) NU pts and existing plan.
   For type 1 and 3: cj is input, fk is output.
   For type 2: fk is input, cj is output.
   Performs spread/interp, pre/post deconvolve, and fftw_execute as appropriate
   for each of the 3 types.
   For cases of num_transforms>1, performs work in blocks of size up to batch_size.
   Return value 0 (no error diagnosis yet).
   Barnett 5/20/20, based on Malleo 2019.
*/
template<typename FloatType>
Status Plan<CPUDevice, FloatType>::execute(DType* cj, DType* fk){
  if (this->type_ != TransformType::TYPE_3) {
    for (int b=0; b*this->batch_size_ < this->num_transforms_; b++) { // .....loop b over batches

      // current batch is either batch_size, or possibly truncated if last one
      int batch_size = std::min(this->num_transforms_ - b*this->batch_size_, this->batch_size_);
      int bB = b*this->batch_size_;         // index of vector, since batchsizes same
      DType* cjb = cj + bB*this->num_points_;        // point to batch of weights
      DType* fkb = fk + bB*this->grid_size_;         // point to batch of mode coeffs

      // STEP 1: (varies by type)
      if (this->type_ == TransformType::TYPE_1) {  // type 1: spread NU pts this->points_[0], weights cj, to fw grid
        TF_RETURN_IF_ERROR(this->spread_or_interp_sorted_batch(batch_size, cjb));
      } else {          //  type 2: amplify Fourier coeffs fk into 0-padded fw
        TF_RETURN_IF_ERROR(this->deconvolve_batch(batch_size, fkb));
      }

      // STEP 2: call the pre-planned FFT on this batch
      // This wastes some flops if batch_size < this->batch_size_.
      fftw::execute<FloatType>(this->fft_plan_);

      // STEP 3: (varies by type)
      if (this->type_ == TransformType::TYPE_1) {   // type 1: deconvolve (amplify) fw and shuffle to fk
        TF_RETURN_IF_ERROR(this->deconvolve_batch(batch_size, fkb));
      } else {          // type 2: interpolate unif fw grid to NU target pts
        TF_RETURN_IF_ERROR(this->spread_or_interp_sorted_batch(batch_size, cjb));
      }
    }                                                   // ........end b loop
  } else {
    // Type 3 transform.
    return errors::Unimplemented("Type-3 transforms not implemented yet.");
  }

  return OkStatus();
}

template<typename FloatType>
Status Plan<CPUDevice, FloatType>::interp(DType* c, DType* f) {
  return this->spread_or_interp(c, f);
}

template<typename FloatType>
Status Plan<CPUDevice, FloatType>::spread(DType* c, DType* f) {
  return this->spread_or_interp(c, f);
}

template<typename FloatType>
Status Plan<CPUDevice, FloatType>::initialize_fft() {
  using FftwType = typename fftw::ComplexType<FloatType>::Type;

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
      fftw::plan_with_nthreads<FloatType>(this->options_.num_threads);
      #endif
      is_fftw_initialized = true;
    }
  }

  // Get FFT dimensions (must be reversed).
  int fft_dims[3] = {1, 1, 1};
  switch (this->rank_) {
    case 1:
      fft_dims[0] = this->fine_dims_[0];
      break;
    case 2:
      fft_dims[1] = this->fine_dims_[0];
      fft_dims[0] = this->fine_dims_[1];
      break;
    case 3:
      fft_dims[2] = this->fine_dims_[0];
      fft_dims[1] = this->fine_dims_[1];
      fft_dims[0] = this->fine_dims_[2];
      break;
  }

  // FFTW flags.
  unsigned flags = 0;
  switch (this->options_.fftw().planning_rigor()) {
    case FftwPlanningRigor::AUTO:       flags = FFTW_MEASURE;     break;
    case FftwPlanningRigor::ESTIMATE:   flags = FFTW_ESTIMATE;    break;
    case FftwPlanningRigor::MEASURE:    flags = FFTW_MEASURE;     break;
    case FftwPlanningRigor::PATIENT:    flags = FFTW_PATIENT;     break;
    case FftwPlanningRigor::EXHAUSTIVE: flags = FFTW_EXHAUSTIVE;  break;
  }

  #pragma omp critical
  {
    this->fft_plan_ = fftw::plan_many_dft<FloatType>(
        /* int rank */ this->rank_,
        /* const int *n */ fft_dims,
        /* int howmany */ this->batch_size_,
        /* fftw_complex *in */ reinterpret_cast<FftwType*>(this->fine_data_),
        /* const int *inembed */ nullptr,
        /* int istride */ 1,
        /* int idist */ this->fine_size_,
        /* fftw_complex *out */ reinterpret_cast<FftwType*>(this->fine_data_),
        /* const int *onembed */ nullptr,
        /* int ostride */ 1,
        /* int odist */ this->fine_size_,
        /* int sign */ static_cast<int>(this->fft_direction_),
        /* unsigned flags */ flags);
  }

  return OkStatus();
}

template<typename FloatType>
Status Plan<CPUDevice, FloatType>::binsort_if_needed() {
  // Allocate memory for bin-sort indices.
  TF_RETURN_IF_ERROR(this->context_->allocate_temp(
      DT_INT32, TensorShape({this->num_points_}),
      &this->binsort_indices_));

  // Heuristics based on cache sizes (only useful for single-thread).
  bool should_sort = !(
      this->rank_ == 1 && (
          this->spread_direction_ == SpreadDirection::INTERP ||
          (this->num_points_ > 1000 * this->fine_size_)));

  if (this->options_.sort_points == SortPoints::YES ||
     (this->options_.sort_points == SortPoints::AUTO && should_sort)) {
    this->do_binsort_ = true;
  } else {
    this->do_binsort_ = false;
  }

  int num_threads = this->options_.num_threads;

  if (this->do_binsort_) {
    // Heuristic: when size of grid is much larger than the number of
    // non-uniform points, single-thread is better.
    if (this->fine_size_ > 10 * this->num_points_) {
      num_threads = 1;
    }
    if (num_threads == 1) {
      this->binsort_singlethread();
    } else {
      this->binsort_multithread();
    }
  } else {
    // Set identity permutation.
    int* binsort_indices = this->binsort_indices_.flat<int>().data();
    thrust::sequence(this->execution_policy(), binsort_indices,
                     binsort_indices + this->num_points_);
  }

  return OkStatus();
}

template<typename FloatType>
void Plan<CPUDevice, FloatType>::binsort_singlethread() {
  // Heuristic for bin dimensions. Affects performance.
  int bin_dims[3] = {16, 4, 4};

  // Number of bins in the box along each dimension.
  int box_dims[3];
  int num_bins = 1;
  for (int d = 0; d < this->rank_; ++d) {
    // +1 needed for robustness against round-off error near +pi.
    // Note that round-off near -pi stably rounds negative to 0.
    box_dims[d] = this->fine_dims_[d] / bin_dims[d] + 1;
    num_bins *= box_dims[d];
  }

  std::vector<int> counts(num_bins, 0);  // count how many pts in each bin
  for (int i = 0; i < this->num_points_; ++i) {
    // find the bin index in however many dims are needed
    int bin_multi_index[3];
    int bin_index = 0;
    for (int d = this->rank_ - 1; d >= 0; --d) {
      bin_multi_index[d] = FOLD_AND_RESCALE(
          this->points_[d][i], this->fine_dims_[d], pirange) / bin_dims[d];
      bin_index = bin_multi_index[d] + box_dims[d] * bin_index;
    }
    counts[bin_index]++;
  }

  // Cumulative sum of bin counts.
  std::vector<int> offsets(num_bins);
  offsets[0] = 0;
  for (int i = 1; i < num_bins; ++i) {
    offsets[i] = offsets[i - 1] + counts[i - 1];
  }

  // Fill inverse map.
  std::vector<int> inv(this->num_points_);
  for (int i = 0; i < this->num_points_; ++i) {
    // find the bin index (again! but better than using RAM)
    int bin_multi_index[3];
    int bin_index = 0;
    for (int d = this->rank_ - 1; d >= 0; --d) {
      bin_multi_index[d] = FOLD_AND_RESCALE(
          this->points_[d][i], this->fine_dims_[d], pirange) / bin_dims[d];
      bin_index = bin_multi_index[d] + box_dims[d] * bin_index;
    }
    int offset = offsets[bin_index];
    offsets[bin_index]++;
    inv[i] = offset;
  }

  // invert the map, writing to output pointer (writing pattern is random)
  int* binsort_indices = this->binsort_indices_.flat<int>().data();
  for (int i = 0; i < this->num_points_; i++) {
    binsort_indices[inv[i]] = i;
  }
}

template<typename FloatType>
void Plan<CPUDevice, FloatType>::binsort_multithread() {
  // Heuristic for bin dimensions. Affects performance.
  int bin_dims[3] = {16, 4, 4};
  int num_threads = this->options_.num_threads;

  // Can't have more threads than points.
  num_threads = std::min(this->num_points_, num_threads);

  int box_dims[3];
  int num_bins = 1;
  for (int d = 0; d < this->rank_; ++d) {
    box_dims[d] = this->fine_dims_[d] / bin_dims[d] + 1;
    num_bins *= box_dims[d];
  }

  // List of start NU point indices per thread.
  std::vector<int> break_points(num_threads + 1);

  // Distribute the NU points to threads.
  for (int thread_index = 0; thread_index <= num_threads; ++thread_index) {
    break_points[thread_index] = static_cast<int>(
        0.5 + this->num_points_ * thread_index / static_cast<double>(num_threads));
  }

  // Global counts, number of points in each bin.
  std::vector<int> counts(num_bins, 0);

  // Offsets per thread, size num_threads * num_bins,
  // initialize to 0 by copying the counts vector.
  std::vector<std::vector<int>> offsets_per_thread(num_threads, counts);
  {
    // Scope for counts_per_thread, the 2D array of counts in bins for each
    // thread's NU points. Has size num_threads * num_bins, init to 0 by
    // copying counts.
    std::vector<std::vector<int>> counts_per_thread(num_threads, counts);

    #pragma omp parallel num_threads(num_threads)
    {
      // Parallel binning to each thread's count. Block done once per thread.
      int thread_index = OMP_GET_THREAD_NUM();

      for (int i = break_points[thread_index];
           i < break_points[thread_index + 1];
           ++i) {
        // Find the bin index in however many dims are needed.
        int bin_multi_index[3];
        int bin_index = 0;
        for (int d = this->rank_ - 1; d >= 0; --d) {
          bin_multi_index[d] = FOLD_AND_RESCALE(
              this->points_[d][i], this->fine_dims_[d], pirange) / bin_dims[d];
          bin_index = bin_multi_index[d] + box_dims[d] * bin_index;
        }
        // Each element only ever accessed by one thread, so read-write op is
        // safe.
        counts_per_thread[thread_index][bin_index]++;
      }
    }

    // Sum along thread axis to get global counts.
    // Not worth parallelizing this loop. Either loop order is OK.
    for (int bin_index = 0; bin_index < num_bins; ++bin_index) {
      for (int thread_index = 0; thread_index < num_threads; ++thread_index) {
	      counts[bin_index] += counts_per_thread[thread_index][bin_index];
      }
    }

    // Cumulative sum of bin counts.
    std::vector<int> offsets(num_bins);
    offsets[0] = 0;
    for (int bin_index = 1; bin_index < num_bins; ++bin_index) {
      offsets[bin_index] = offsets[bin_index - 1] + counts[bin_index - 1];
    }

    // Now build offsets for each thread and bin.
    for (int bin_index = 0; bin_index < num_bins; ++bin_index) {
      offsets_per_thread[0][bin_index] = offsets[bin_index];
    }
    // Again not worth parallelizing. Either loop order OK.
    for (int thread_index = 1; thread_index < num_threads; ++thread_index) {
      for (int bin_index = 0; bin_index < num_bins; ++bin_index) {
        // Cumulative sum along threads axis.
	      offsets_per_thread[thread_index][bin_index] = (
            offsets_per_thread[thread_index - 1][bin_index] +
            counts_per_thread[thread_index - 1][bin_index]);
      }
    }
  }  // Scope frees up counts_per_thread here.

  // Fill inverse map, in parallel.
  std::vector<int> inv(this->num_points_);
  #pragma omp parallel num_threads(num_threads)
  {
    int thread_index = OMP_GET_THREAD_NUM();
    for (int i = break_points[thread_index];
         i < break_points[thread_index + 1];
         ++i) {
      // Find the bin index (again, but better than using RAM).
      int bin_multi_index[3];
      int bin_index = 0;
      for (int d = this->rank_ - 1; d >= 0; --d) {
        bin_multi_index[d] = FOLD_AND_RESCALE(
            this->points_[d][i], this->fine_dims_[d], pirange) / bin_dims[d];
        bin_index = bin_multi_index[d] + box_dims[d] * bin_index;
      }
      // Get the offset for this NU point and thread.
      inv[i] = offsets_per_thread[thread_index][bin_index];
      // Each element only ever accessed by one thread, so read-write op is
      // safe.
      offsets_per_thread[thread_index][bin_index]++;
    }
  }

  // Invert the map.
  int* binsort_indices = this->binsort_indices_.flat<int>().data();
  #pragma omp parallel for num_threads(num_threads) schedule(dynamic,10000)
  for (int i = 0; i < this->num_points_; ++i) {
    binsort_indices[inv[i]] = i;
  }
}

template<typename FloatType>
Status Plan<CPUDevice, FloatType>::spread_or_interp(DType* cj, DType* fk) {
  // Loop over batches.
  for (int batch_index = 0;
       batch_index * this->batch_size_ < this->num_transforms_;
       batch_index++) {

    // Get current batch size (possibly truncated if last one).
    int batch_size = std::min(
        this->num_transforms_ - batch_index * this->batch_size_,
        this->batch_size_);

    // Execute this batch.
    TF_RETURN_IF_ERROR(this->spread_or_interp_sorted_batch(
        batch_size,
        cj + batch_index * this->batch_size_ * this->num_points_,
        fk + batch_index * this->batch_size_ * this->grid_size_));
  }

  return OkStatus();
}

template<typename FloatType>
Status Plan<CPUDevice, FloatType>::spread_or_interp_sorted_batch(
    int batch_size, DType* cBatch, DType* fBatch) {
  // opts.spread_threading: 1 sequential multithread, 2 parallel single-thread.
  // omp_sets_nested deprecated, so don't use; assume not nested for 2 to work.
  // But when nthr_outer=1 here, omp par inside the loop sees all threads...
  int nthr_outer = this->options_.spread_threading == SpreadThreading::SEQUENTIAL_MULTI_THREADED ? 1 : batch_size;

  if (fBatch == nullptr) {
    fBatch = (DType*) this->fine_data_;
  }

  int64_t grid_size_0 = this->fine_dims_[0];
  int64_t grid_size_1 = 1;
  int64_t grid_size_2 = 1;
  if (this->rank_ > 1) grid_size_1 = this->fine_dims_[1];
  if (this->rank_ > 2) grid_size_2 = this->fine_dims_[2];

  #pragma omp parallel for num_threads(nthr_outer)
  for (int i=0; i<batch_size; i++) {
    DType *fwi = fBatch + i*this->fine_size_;  // start of i'th fw array in wkspace
    DType *ci = cBatch + i*this->num_points_;            // start of i'th c array in cBatch
    spreadinterpSorted(this->binsort_indices_.flat<int>().data(), grid_size_0, grid_size_1, grid_size_2,
                       (FloatType*)fwi, this->num_points_, this->points_[0], this->points_[1], this->points_[2],
                       (FloatType*)ci, this->spread_params_, this->do_binsort_);
  }
  return OkStatus();
}

template<typename FloatType>
Status Plan<CPUDevice, FloatType>::deconvolve_batch(int batch_size, DType* fkBatch) {
  #pragma omp parallel for num_threads(batch_size)
  for (int elem_index = 0; elem_index < batch_size; elem_index++) {
    DType *fwi = this->fine_data_ + elem_index * this->fine_size_;
    DType *fki = fkBatch + elem_index * this->grid_size_;
    switch (this->rank_) {
      case 1: {
        this->deconvolve_1d(fki, fwi);
        break;
      }
      case 2: {
        this->deconvolve_2d(fki, fwi);
        break;
      }
      case 3: {
        this->deconvolve_3d(fki, fwi);
        break;
      }
    }
  }
  return OkStatus();
}

template<typename FloatType>
void Plan<CPUDevice, FloatType>::deconvolve_1d(
    DType *fk, DType* fw, FloatType prefactor) {

  int64_t kmin = -this->grid_dims_[0] / 2;
  int64_t kmax = (this->grid_dims_[0] - 1) / 2;

  if (this->grid_dims_[0] == 0) {
    kmax = -1;
  }

  // Get start of positive and negative chunks of fk array.
  int64_t pp, pn;
  switch (this->options_.mode_order) {
    case ModeOrder::FFT: {
      pp = 0;
      pn = kmax + 1;
      break;
    }
    case ModeOrder::CMCL: {
      pp = -kmin;
      pn = 0;
      break;
    }
  }

  if (this->spread_params_.spread_direction == SpreadDirection::SPREAD) {
    // Non-negative frequencies.
    for (int64_t k = 0; k <= kmax; ++k) {
      fk[pp++] = prefactor * fw[k] / this->fseries_data_[0][k];
    }
    // Negative frequencies.
    for (int64_t k = kmin; k < 0; ++k) {
      fk[pn++] = prefactor * fw[this->fine_dims_[0] + k] /
          this->fseries_data_[0][-k];
    }
  } else {
    // Zero-padding.
    for (int64_t k = kmax + 1; k < this->fine_dims_[0] + kmin; ++k) {
      fw[k] = std::complex(0.0, 0.0);
    }
    // Non-negative frequencies.
    for (int64_t k = 0;k <= kmax; ++k) {
      fw[k] = prefactor * fk[pp++] / this->fseries_data_[0][k];
    }
    // Negative frequencies.
    for (int64_t k = kmin; k < 0; ++k) {
      fw[this->fine_dims_[0] + k] =
          prefactor * fk[pn++] / this->fseries_data_[0][-k];
    }
  }
}

template<typename FloatType>
void Plan<CPUDevice, FloatType>::deconvolve_2d(
    DType *fk, DType* fw, FloatType prefactor) {
  FloatType* ker2 = this->fseries_data_[1];
  int64_t ms = this->grid_dims_[0];
  int64_t mt = this->grid_dims_[1];
  int64_t nf1 = this->fine_dims_[0];
  int64_t nf2 = this->fine_dims_[1];

  int64_t k2min = -mt / 2;
  int64_t k2max = (mt - 1) / 2;

  if (mt == 0) {
    k2max = -1;
  }

  // Get start of positive and negative chunks of fk array.
  int64_t pp, pn;
  switch (this->options_.mode_order) {
    case ModeOrder::FFT: {
      pp = 0;
      pn = (k2max + 1) * ms;
      break;
    }
    case ModeOrder::CMCL: {
      pp = -k2min * ms;
      pn = 0;
      break;
    }
  }

  // Zero-padding.
  if (this->spread_params_.spread_direction == SpreadDirection::INTERP) {
    for (int64_t j = nf1 * (k2max + 1); j < nf1 * (nf2 + k2min); ++j)  {
      fw[j] = std::complex(0.0, 0.0);
    }
  }

  // Non-negative frequencies.
  for (int64_t k2 = 0; k2 <= k2max; ++k2, pp += ms) {
    this->deconvolve_1d(fk + pp, &fw[nf1 * k2], prefactor / ker2[k2]);
  }

  // Negative frequencies.
  for (int64_t k2 = k2min; k2 < 0; ++k2, pn += ms) {
    this->deconvolve_1d(fk + pn, &fw[nf1 * (nf2 + k2)], prefactor / ker2[-k2]);
  }
}

template<typename FloatType>
void Plan<CPUDevice, FloatType>::deconvolve_3d(
    DType *fk, DType* fw, FloatType prefactor) {
  FloatType* ker3 = this->fseries_data_[2];
  int64_t ms = this->grid_dims_[0];
  int64_t mt = this->grid_dims_[1];
  int64_t mu = this->grid_dims_[2];
  int64_t nf1 = this->fine_dims_[0];
  int64_t nf2 = this->fine_dims_[1];
  int64_t nf3 = this->fine_dims_[2];

  int64_t k3min = -mu / 2;
  int64_t k3max = (mu - 1) / 2;
  if (mu == 0) {
    k3max = -1;
  }

  // Get start of positive and negative chunks of fk array.
  int64_t pp, pn;
  switch (this->options_.mode_order) {
    case ModeOrder::FFT: {
      pp = 0;
      pn = (k3max + 1) * ms * mt;
      break;
    }
    case ModeOrder::CMCL: {
      pp = -k3min * ms * mt;
      pn = 0;
      break;
    }
  }

  int64_t np = nf1 * nf2;

  // Zero-padding.
  if (this->spread_params_.spread_direction == SpreadDirection::INTERP) {
    for (int64_t j = np * (k3max + 1); j < np * (nf3 + k3min); ++j) {
      fw[j] = std::complex(0.0, 0.0);
    }
  }

  // Non-negative frequencies.
  for (int64_t k3 = 0; k3 <= k3max; ++k3, pp += ms * mt) {
    this->deconvolve_2d(fk + pp, &fw[np * k3], prefactor / ker3[k3]);
  }

  // Negative frequencies.
  for (int64_t k3 = k3min; k3 < 0; ++k3, pn += ms * mt) {
    this->deconvolve_2d(fk + pn, &fw[np * (nf3 + k3)], prefactor / ker3[-k3]);
  }
}

namespace {

template<typename FloatType>
Status setup_spreader(
    int rank,
    int kerevalmeth, bool show_warnings,
    const InternalOptions& options,
    SpreadParameters<FloatType> &spread_params)
/* Initializes spreader kernel parameters given desired NUFFT tol eps,
   upsampling factor (=sigma in paper, or R in Dutt-Rokhlin), ker eval meth
   (either 0:exp(sqrt()), 1: Horner ppval), and some debug-level flags.
   Also sets all default options in SpreadParameters<FloatType>. See SpreadParameters<FloatType>.h for spread_params.
   rank is spatial dimension (1,2, or 3).
   See finufft.cpp:finufft_plan() for where upsampling_factor is set.
   Must call this before any kernel evals done, otherwise segfault likely.
   Barnett 2017. debug, loosened eps logic 6/14/20.
*/
{
  if (options.upsampling_factor != 2.0 && options.upsampling_factor != 1.25) {
    if (kerevalmeth == 1) {
      return errors::Internal(
          "Horner kernel evaluation only supports standard "
          "upsampling factors of 2.0 or 1.25, but got ",
          options.upsampling_factor);
    }
  }

  // write out default SpreadParameters<FloatType>
  spread_params.pirange = 1;             // user also should always set this
  spread_params.sort_points = SortPoints::AUTO;
  spread_params.pad_kernel = 0;              // affects only evaluate_kernel_vector
  spread_params.kerevalmeth = kerevalmeth;
  spread_params.upsampling_factor = options.upsampling_factor;
  spread_params.num_threads = 0;            // all avail
  spread_params.sort_threads = 0;        // 0:auto-choice
  // heuristic dir=1 chunking for num_threads>>1, typical for intel i7 and skylake...
  spread_params.max_subproblem_size = (rank == 1) ? 10000 : 100000;
  spread_params.flags = 0;               // 0:no timing flags (>0 for experts only)
  spread_params.verbosity = 0;               // 0:no debug output
  // heuristic num_threads above which switch OMP critical to atomic (add_wrapped...):
  spread_params.atomic_threshold = 10;   // R Blackwell's value

  int ns = options.kernel_width;
  spread_params.kernel_width = ns;

  // setup for reference kernel eval (via formula): select beta width param...
  // (even when kerevalmeth=1, this ker eval needed for FTs in onedim_*_kernel)
  spread_params.kernel_half_width = (FloatType)ns / 2;   // constants to help (see below routines)
  spread_params.kernel_c = 4.0 / (FloatType)(ns * ns);
  FloatType beta_over_ns = 2.30;         // gives decent betas for default sigma=2.0
  if (ns == 2) beta_over_ns = 2.20;  // some small-width tweaks...
  if (ns == 3) beta_over_ns = 2.26;
  if (ns == 4) beta_over_ns = 2.38;
  if (options.upsampling_factor != 2.0) {          // again, override beta for custom sigma
    FloatType gamma = 0.97;              // must match devel/gen_all_horner_C_code.m !
    beta_over_ns = gamma * kPi<FloatType>*(1.0 - 1.0 / (2 * options.upsampling_factor));  // formula based on cutoff
  }
  spread_params.kernel_beta = beta_over_ns * (FloatType)ns;    // set the kernel beta parameter

  // Calculate scaling factor for spread/interp only mode.
  if (spread_params.spread_only)
    spread_params.kernel_scale = calculate_scale_factor<FloatType>(rank, spread_params);

  return OkStatus();
}

template<typename FloatType>
Status setup_spreader_for_nufft(int rank,
                                const InternalOptions& options,
                                SpreadParameters<FloatType> &spread_params)
// Set up the spreader parameters given eps, and pass across various nufft
// options. Return status of setup_spreader. Uses pass-by-ref. Barnett 10/30/17
{
  // This must be set before calling setup_spreader
  spread_params.spread_only = options.spread_only;

  TF_RETURN_IF_ERROR(setup_spreader(
      rank,
      static_cast<int>(options.kernel_evaluation_method) - 1, // We subtract 1 temporarily, as spreader expects values of 0 or 1 instead of 1 and 2.
      options.show_warnings, options, spread_params));

  // override various spread spread_params from their defaults...
  spread_params.sort_points = options.sort_points;
  spread_params.spread_method = options.spread_method;
  spread_params.verbosity = options.verbosity;
  spread_params.pad_kernel = options.pad_kernel; // (only applies to kerevalmeth=0)
  spread_params.num_threads = options.num_threads;
  if (options.num_threads_for_atomic_spread >= 0) // overrides
    spread_params.atomic_threshold = options.num_threads_for_atomic_spread;
  if (options.max_spread_subproblem_size > 0)        // overrides
    spread_params.max_subproblem_size = options.max_spread_subproblem_size;

  return OkStatus();
}

// Checks spreader inputs.
// Split out by Melody Shih, Jun 2018. Finiteness chk Barnett 7/30/18.
// Bypass FOLD_AND_RESCALE macro which has inevitable rounding err even nr +pi,
// giving fake invalids well inside the [-3pi,3pi] domain, 4/9/21.
// Equivalent to spreadcheck in original FINUFFT code.
template<typename FloatType>
Status check_spread_inputs(int64_t n1, int64_t n2, int64_t n3, int64_t num_points,
                           FloatType *kx, FloatType *ky, FloatType *kz,
                           SpreadParameters<FloatType> opts) {
  // Check that cuboid is large enough for spreading.
  int min_n = 2 * opts.kernel_width;
  if (n1 < min_n || (n2 > 1 && n2 < min_n) || (n3 > 1 && n3 < min_n)) {
    return errors::InvalidArgument(
        "cuboid too small for spreading, got (", n1, ", ", n2, ", ", n3, ") ",
        "but need at least ", min_n, " in each non-trivial dimension");
  }

  return OkStatus();
}


static int get_transform_rank(int64_t n1, int64_t n2, int64_t n3) {
  int rank = 1;
  if (n2 > 1) ++rank;
  if (n3 > 1) ++rank;
  return rank;
}


template<typename FloatType>
int spreadinterpSorted(int* sort_indices, int64_t N1, int64_t N2, int64_t N3,
		      FloatType *data_uniform, int64_t M, FloatType *kx, FloatType *ky, FloatType *kz,
		      FloatType *data_nonuniform, SpreadParameters<FloatType> opts, int did_sort)
/* Logic to select the main spreading (dir=1) vs interpolation (dir=2) routine.
   See spreadinterp() above for inputs arguments and definitions.
   Return value should always be 0 (no error reporting).
   Split out by Melody Shih, Jun 2018; renamed Barnett 5/20/20.
*/
{
  if (opts.spread_direction == SpreadDirection::SPREAD)
    spreadSorted(sort_indices, N1, N2, N3, data_uniform, M, kx, ky, kz, data_nonuniform, opts, did_sort);
  else // if (opts.spread_direction == SpreadDirection::INTERP)
    interpSorted(sort_indices, N1, N2, N3, data_uniform, M, kx, ky, kz, data_nonuniform, opts, did_sort);

  return 0;
}


// --------------------------------------------------------------------------
template<typename FloatType>
int spreadSorted(int* sort_indices,int64_t N1, int64_t N2, int64_t N3,
		      FloatType *data_uniform,int64_t M, FloatType *kx, FloatType *ky, FloatType *kz,
		      FloatType *data_nonuniform, SpreadParameters<FloatType> opts, int did_sort)
// Spread NU pts in sorted order to a uniform grid. See spreadinterp() for doc.
{
  int ndims = get_transform_rank(N1,N2,N3);
  int64_t N=N1*N2*N3;            // output array size
  int ns=opts.kernel_width;          // abbrev. for w, kernel width
  int nthr = OMP_GET_MAX_THREADS();  // # threads to use to spread
  if (opts.num_threads>0)
    nthr = std::min(nthr,opts.num_threads);     // user override up to max avail

  for (int64_t i=0; i<2*N; i++) // zero the output array. std::fill is no faster
    data_uniform[i]=0.0;

  // If there are no non-uniform points, we're done.
  if (M == 0) return 0;

  int spread_single = (nthr==1) || (M*100<N);     // low-density heuristic?
  spread_single = 0;                 // for now
  if (spread_single) {    // ------- Basic single-core t1 spreading ------
    for (int64_t j=0; j<M; j++) {
      // *** todo, not urgent
      // ... (question is: will the index wrapping per NU pt slow it down?)
    }

  } else {           // ------- Fancy multi-core blocked t1 spreading ----
                     // Splits sorted inds (jfm's advanced2), could double RAM.
    // choose nb (# subprobs) via used num_threads:
    int nb = std::min((int64_t)nthr,M);         // simply split one subprob per thr...
    if (nb*(int64_t)opts.max_subproblem_size<M) {  // ...or more subprobs to cap size
      nb = 1 + (M-1)/opts.max_subproblem_size;  // int div does ceil(M/opts.max_subproblem_size)
      if (opts.verbosity) printf("\tcapping subproblem sizes to max of %d\n",opts.max_subproblem_size);
    }
    if (M*1000<N) {         // low-density heuristic: one thread per NU pt!
      nb = M;
      if (opts.verbosity) printf("\tusing low-density speed rescue nb=M...\n");
    }
    if (!did_sort && nthr==1) {
      nb = 1;
      if (opts.verbosity) printf("\tunsorted nthr=1: forcing single subproblem...\n");
    }

    std::vector<int64_t> brk(nb+1); // NU index breakpoints defining nb subproblems
    for (int p = 0; p <= nb; ++p)
      brk[p] = (int64_t)(0.5 + M * p / (double)nb);

    #pragma omp parallel for num_threads(nthr) schedule(dynamic,1)  // each is big
    for (int isub=0; isub<nb; isub++) {   // Main loop through the subproblems
      int64_t M0 = brk[isub+1]-brk[isub];  // # NU pts in this subproblem
      // copy the location and data vectors for the nonuniform points
      FloatType *kx0=(FloatType*)malloc(sizeof(FloatType)*M0), *ky0=nullptr, *kz0=nullptr;
      if (N2>1)
        ky0=(FloatType*)malloc(sizeof(FloatType)*M0);
      if (N3>1)
        kz0=(FloatType*)malloc(sizeof(FloatType)*M0);
      FloatType *dd0=(FloatType*)malloc(sizeof(FloatType)*M0*2);    // complex strength data
      for (int64_t j=0; j<M0; j++) {           // todo: can avoid this copying?
        int64_t kk=sort_indices[j+brk[isub]];  // NU pt from subprob index list
        kx0[j]=FOLD_AND_RESCALE(kx[kk],N1,opts.pirange);
        if (N2>1) ky0[j]=FOLD_AND_RESCALE(ky[kk],N2,opts.pirange);
        if (N3>1) kz0[j]=FOLD_AND_RESCALE(kz[kk],N3,opts.pirange);
        dd0[j*2]=data_nonuniform[kk*2];     // real part
        dd0[j*2+1]=data_nonuniform[kk*2+1]; // imag part
      }
      // get the subgrid which will include padding by roughly kernel_width/2
      int64_t offset1,offset2,offset3,size1,size2,size3; // get_subgrid sets
      get_subgrid(offset1,offset2,offset3,size1,size2,size3,M0,kx0,ky0,kz0,ns,ndims);  // sets offsets and sizes

      // allocate output data for this subgrid
      FloatType *du0=(FloatType*)malloc(sizeof(FloatType)*2*size1*size2*size3); // complex

      // Spread to subgrid without need for bounds checking or wrapping
      if (ndims==1)
        spread_subproblem_1d(offset1,size1,du0,M0,kx0,dd0,opts);
      else if (ndims==2)
        spread_subproblem_2d(offset1,offset2,size1,size2,du0,M0,kx0,ky0,dd0,opts);
      else
        spread_subproblem_3d(offset1,offset2,offset3,size1,size2,size3,du0,M0,kx0,ky0,kz0,dd0,opts);

      // do the adding of subgrid to output
      if (nthr > opts.atomic_threshold)   // see above for debug reporting
        add_wrapped_subgrid_thread_safe(offset1,offset2,offset3,size1,size2,size3,N1,N2,N3,data_uniform,du0);   // R Blackwell's atomic version
      else {
        #pragma omp critical
        add_wrapped_subgrid(offset1,offset2,offset3,size1,size2,size3,N1,N2,N3,data_uniform,du0);
      }

      // free up stuff from this subprob... (that was malloc'ed by hand)
      free(dd0);
      free(du0);
      free(kx0);
      if (N2 > 1) free(ky0);
      if (N3 > 1) free(kz0);
    }     // end main loop over subprobs
  }   // end of choice of which t1 spread type to use

  // in spread/interp only mode, apply scaling factor (Montalt 6/8/2021).
  if (opts.spread_only) {
    for (int64_t i = 0; i < 2*N; i++)
      data_uniform[i] *= opts.kernel_scale;
  }

  return 0;
};


// --------------------------------------------------------------------------
template<typename FloatType>
int interpSorted(int* sort_indices,int64_t N1, int64_t N2, int64_t N3,
		      FloatType *data_uniform,int64_t M, FloatType *kx, FloatType *ky, FloatType *kz,
		      FloatType *data_nonuniform, SpreadParameters<FloatType> opts, int did_sort)
// Interpolate to NU pts in sorted order from a uniform grid.
// See spreadinterp() for doc.
{
  int ndims = get_transform_rank(N1,N2,N3);
  int ns=opts.kernel_width;          // abbrev. for w, kernel width
  FloatType ns2 = (FloatType)ns/2;          // half spread width, used as stencil shift
  int nthr = OMP_GET_MAX_THREADS();   // # threads to use to interp
  if (opts.num_threads > 0)
    nthr = std::min(nthr, opts.num_threads);

  #pragma omp parallel num_threads(nthr)
  {
    #define CHUNK_SIZE 16     // Chunks of Type 2 targets (Ludvig found by expt)
    int64_t jlist[CHUNK_SIZE];
    FloatType xjlist[CHUNK_SIZE], yjlist[CHUNK_SIZE], zjlist[CHUNK_SIZE];
    FloatType outbuf[2 * CHUNK_SIZE];
    // Kernels: static alloc is faster, so we do it for up to 3D...
    FloatType kernel_args[3 * MAX_KERNEL_WIDTH];
    FloatType kernel_values[3 * MAX_KERNEL_WIDTH];
    FloatType *ker1 = kernel_values;
    FloatType *ker2 = kernel_values + ns;
    FloatType *ker3 = kernel_values + 2 * ns;

    // Loop over interpolation chunks
    #pragma omp for schedule (dynamic,1000)  // assign threads to NU targ pts:
    for (int64_t i=0; i<M; i+=CHUNK_SIZE) { // main loop over NU targs, interp each from U
      // Setup buffers for this chunk
      int bufsize = (i+CHUNK_SIZE > M) ? M-i : CHUNK_SIZE;
      for (int ibuf=0; ibuf<bufsize; ibuf++) {
        int64_t j = sort_indices[i+ibuf];
        jlist[ibuf] = j;
        xjlist[ibuf] = FOLD_AND_RESCALE(kx[j],N1,opts.pirange);
        if(ndims >=2)
          yjlist[ibuf] = FOLD_AND_RESCALE(ky[j],N2,opts.pirange);
        if(ndims == 3)
          zjlist[ibuf] = FOLD_AND_RESCALE(kz[j],N3,opts.pirange);
      }

      // Loop over targets in chunk
      for (int ibuf=0; ibuf<bufsize; ibuf++) {
        FloatType xj = xjlist[ibuf];
        FloatType yj = (ndims > 1) ? yjlist[ibuf] : 0;
        FloatType zj = (ndims > 2) ? zjlist[ibuf] : 0;

        FloatType *target = outbuf+2*ibuf;

        // coords (x,y,z), spread block corner index (i1,i2,i3) of current NU targ
        int64_t i1=(int64_t)std::ceil(xj-ns2); // leftmost grid index
        int64_t i2= (ndims > 1) ? (int64_t)std::ceil(yj-ns2) : 0; // min y grid index
        int64_t i3= (ndims > 1) ? (int64_t)std::ceil(zj-ns2) : 0; // min z grid index

        FloatType x1=(FloatType)i1-xj;           // shift of ker center, in [-w/2,-w/2+1]
        FloatType x2= (ndims > 1) ? (FloatType)i2-yj : 0 ;
        FloatType x3= (ndims > 2)? (FloatType)i3-zj : 0;

        // eval kernel values patch and use to interpolate from uniform data...
        if (opts.kerevalmeth==0) {               // choose eval method
          set_kernel_args(kernel_args, x1, opts);
          if(ndims > 1)  set_kernel_args(kernel_args+ns, x2, opts);
          if(ndims > 2)  set_kernel_args(kernel_args+2*ns, x3, opts);

          evaluate_kernel_vector(kernel_values, kernel_args, opts, ndims*ns);
        } else {
          eval_kernel_vec_Horner(ker1,x1,ns,opts);
          if (ndims > 1) eval_kernel_vec_Horner(ker2,x2,ns,opts);
          if (ndims > 2) eval_kernel_vec_Horner(ker3,x3,ns,opts);
        }

        switch (ndims) {
          case 1:
            interp_line(target,data_uniform,ker1,i1,N1,ns);
            break;
          case 2:
            interp_square(target,data_uniform,ker1,ker2,i1,i2,N1,N2,ns);
            break;
          case 3:
            interp_cube(target,data_uniform,ker1,ker2,ker3,i1,i2,i3,N1,N2,N3,ns);
            break;
          default: //can't get here
            break;
        }

        // in spread/interp only mode, apply scaling factor (Montalt 6/8/2021).
        if (opts.spread_only) {
          target[0] *= opts.kernel_scale;
          target[1] *= opts.kernel_scale;
        }
      }  // end loop over targets in chunk

      // Copy result buffer to output array
      for (int ibuf=0; ibuf<bufsize; ibuf++) {
        int64_t j = jlist[ibuf];
        data_nonuniform[2*j] = outbuf[2*ibuf];
        data_nonuniform[2*j+1] = outbuf[2*ibuf+1];
      }

    }  // end NU targ loop
  }  // end parallel section

  return 0;
};

///////////////////////////////////////////////////////////////////////////

template<typename FloatType>
static inline void set_kernel_args(FloatType *args, FloatType x, const SpreadParameters<FloatType>& opts)
// Fills vector args[] with kernel arguments x, x+1, ..., x+ns-1.
// needed for the vectorized kernel eval of Ludvig af K.
{
  int ns=opts.kernel_width;
  for (int i=0; i<ns; i++)
    args[i] = x + (FloatType) i;
}

template<typename FloatType>
static inline void evaluate_kernel_vector(FloatType *ker, FloatType *args, const SpreadParameters<FloatType>& opts, const int N)
/* Evaluate ES kernel for a vector of N arguments; by Ludvig af K.
   If opts.pad_kernel true, args and ker must be allocated for Npad, and args is
   written to (to pad to length Npad), only first N outputs are correct.
   Barnett 4/24/18 option to pad to mult of 4 for better SIMD vectorization.

   Obsolete (replaced by Horner), but keep around for experimentation since
   works for arbitrary beta. Formula must match reference implementation. */
{
  FloatType b = opts.kernel_beta;
  FloatType c = opts.kernel_c;

  // Note (by Ludvig af K): Splitting kernel evaluation into two loops
  // seems to benefit auto-vectorization.
  // gcc 5.4 vectorizes first loop; gcc 7.2 vectorizes both loops
  int Npad = N;
  if (opts.pad_kernel) {        // since always same branch, no speed hit
    Npad = 4*(1+(N-1)/4);   // pad N to mult of 4; help i7 GCC, not xeon
    for (int i=N;i<Npad;++i)    // pad with 1-3 zeros for safe eval
      args[i] = 0.0;
  }

  // Loop 1: Compute exponential arguments.
  for (int i = 0; i < Npad; i++) {
    ker[i] = b * sqrt(1.0 - c * args[i] * args[i]);
  }
  // Loop 2: Compute exponentials.
  for (int i = 0; i < Npad; i++) {
  	ker[i] = exp(ker[i]);
  }
  // Separate check from arithmetic (Is this really needed? doesn't slow down)
  for (int i = 0; i < N; i++) {
    if (abs(args[i])>=opts.kernel_half_width) ker[i] = 0.0;
  }
}

template<typename FloatType>
static inline void eval_kernel_vec_Horner(FloatType *ker, const FloatType x, const int w,
					  const SpreadParameters<FloatType> &opts)
/* Fill ker[] with Horner piecewise poly approx to [-w/2,w/2] ES kernel eval at
   x_j = x + j,  for j=0,..,w-1.  Thus x in [-w/2,-w/2+1].   w is aka ns.
   This is the current evaluation method, since it's faster (except i7 w=16).
   Two upsampfacs implemented. Params must match ref formula. Barnett 4/24/18 */
{
  FloatType z = 2 * x + w - 1.0;         // scale so local grid offset z in [-1,1]
  // insert the auto-generated code which expects z, w args, writes to ker...
  if (opts.upsampling_factor == 2.0) {     // floating point equality is fine here
    #include "kernel_horner_sigma2.inc"
  } else if (opts.upsampling_factor == 1.25) {
    #include "kernel_horner_sigma125.inc"
  } else
    fprintf(stderr,"%s: unknown upsampling_factor, failed!\n",__func__);
}

template<typename FloatType>
void interp_line(FloatType *target,FloatType *du, FloatType *ker,int64_t i1,int64_t N1,int ns)
// 1D interpolate complex values from du array to out, using real weights
// ker[0] through ker[ns-1]. out must be size 2 (real,imag), and du
// of size 2*N1 (alternating real,imag). i1 is the left-most index in [0,N1)
// Periodic wrapping in the du array is applied, assuming N1>=ns.
// dx is index into ker array, j index in complex du (data_uniform) array.
// Barnett 6/15/17
{
  FloatType out[] = {0.0, 0.0};
  int64_t j = i1;
  if (i1<0) {                               // wraps at left
    j+=N1;
    for (int dx=0; dx<-i1; ++dx) {
      out[0] += du[2*j]*ker[dx];
      out[1] += du[2*j+1]*ker[dx];
      ++j;
    }
    j-=N1;
    for (int dx=-i1; dx<ns; ++dx) {
      out[0] += du[2*j]*ker[dx];
      out[1] += du[2*j+1]*ker[dx];
      ++j;
    }
  } else if (i1+ns>=N1) {                    // wraps at right
    for (int dx=0; dx<N1-i1; ++dx) {
      out[0] += du[2*j]*ker[dx];
      out[1] += du[2*j+1]*ker[dx];
      ++j;
    }
    j-=N1;
    for (int dx=N1-i1; dx<ns; ++dx) {
      out[0] += du[2*j]*ker[dx];
      out[1] += du[2*j+1]*ker[dx];
      ++j;
    }
  } else {                                     // doesn't wrap
    for (int dx=0; dx<ns; ++dx) {
      out[0] += du[2*j]*ker[dx];
      out[1] += du[2*j+1]*ker[dx];
      ++j;
    }
  }
  target[0] = out[0];
  target[1] = out[1];
}

template<typename FloatType>
void interp_square(FloatType *target,FloatType *du, FloatType *ker1, FloatType *ker2, int64_t i1,int64_t i2,int64_t N1,int64_t N2,int ns)
// 2D interpolate complex values from du (uniform grid data) array to out value,
// using ns*ns square of real weights
// in ker. out must be size 2 (real,imag), and du
// of size 2*N1*N2 (alternating real,imag). i1 is the left-most index in [0,N1)
// and i2 the bottom index in [0,N2).
// Periodic wrapping in the du array is applied, assuming N1,N2>=ns.
// dx,dy indices into ker array, j index in complex du array.
// Barnett 6/16/17
{
  FloatType out[] = {0.0, 0.0};
  if (i1>=0 && i1+ns<=N1 && i2>=0 && i2+ns<=N2) {  // no wrapping: avoid ptrs
    for (int dy=0; dy<ns; dy++) {
      int64_t j = N1*(i2+dy) + i1;
      for (int dx=0; dx<ns; dx++) {
	FloatType k = ker1[dx]*ker2[dy];
	out[0] += du[2*j] * k;
	out[1] += du[2*j+1] * k;
	++j;
      }
    }
  } else {                         // wraps somewhere: use ptr list (slower)
    int64_t j1[MAX_KERNEL_WIDTH], j2[MAX_KERNEL_WIDTH];   // 1d ptr lists
    int64_t x=i1, y=i2;                 // initialize coords
    for (int d=0; d<ns; d++) {         // set up ptr lists
      if (x<0) x+=N1;
      if (x>=N1) x-=N1;
      j1[d] = x++;
      if (y<0) y+=N2;
      if (y>=N2) y-=N2;
      j2[d] = y++;
    }
    for (int dy=0; dy<ns; dy++) {      // use the pts lists
      int64_t oy = N1*j2[dy];           // offset due to y
      for (int dx=0; dx<ns; dx++) {
	FloatType k = ker1[dx]*ker2[dy];
	int64_t j = oy + j1[dx];
	out[0] += du[2*j] * k;
	out[1] += du[2*j+1] * k;
      }
    }
  }
  target[0] = out[0];
  target[1] = out[1];
}

template<typename FloatType>
void interp_cube(FloatType *target,FloatType *du, FloatType *ker1, FloatType *ker2, FloatType *ker3,
		 int64_t i1,int64_t i2,int64_t i3, int64_t N1,int64_t N2,int64_t N3,int ns)
// 3D interpolate complex values from du (uniform grid data) array to out value,
// using ns*ns*ns cube of real weights
// in ker. out must be size 2 (real,imag), and du
// of size 2*N1*N2*N3 (alternating real,imag). i1 is the left-most index in
// [0,N1), i2 the bottom index in [0,N2), i3 lowest in [0,N3).
// Periodic wrapping in the du array is applied, assuming N1,N2,N3>=ns.
// dx,dy,dz indices into ker array, j index in complex du array.
// Barnett 6/16/17
{
  FloatType out[] = {0.0, 0.0};
  if (i1>=0 && i1+ns<=N1 && i2>=0 && i2+ns<=N2 && i3>=0 && i3+ns<=N3) {
    // no wrapping: avoid ptrs
    for (int dz=0; dz<ns; dz++) {
      int64_t oz = N1*N2*(i3+dz);        // offset due to z
      for (int dy=0; dy<ns; dy++) {
	int64_t j = oz + N1*(i2+dy) + i1;
	FloatType ker23 = ker2[dy]*ker3[dz];
	for (int dx=0; dx<ns; dx++) {
	  FloatType k = ker1[dx]*ker23;
	  out[0] += du[2*j] * k;
	  out[1] += du[2*j+1] * k;
	  ++j;
	}
      }
    }
  } else {                         // wraps somewhere: use ptr list (slower)
    int64_t j1[MAX_KERNEL_WIDTH], j2[MAX_KERNEL_WIDTH], j3[MAX_KERNEL_WIDTH];   // 1d ptr lists
    int64_t x=i1, y=i2, z=i3;         // initialize coords
    for (int d=0; d<ns; d++) {          // set up ptr lists
      if (x<0) x+=N1;
      if (x>=N1) x-=N1;
      j1[d] = x++;
      if (y<0) y+=N2;
      if (y>=N2) y-=N2;
      j2[d] = y++;
      if (z<0) z+=N3;
      if (z>=N3) z-=N3;
      j3[d] = z++;
    }
    for (int dz=0; dz<ns; dz++) {             // use the pts lists
      int64_t oz = N1*N2*j3[dz];               // offset due to z
      for (int dy=0; dy<ns; dy++) {
	int64_t oy = oz + N1*j2[dy];           // offset due to y & z
	FloatType ker23 = ker2[dy]*ker3[dz];
	for (int dx=0; dx<ns; dx++) {
	  FloatType k = ker1[dx]*ker23;
	  int64_t j = oy + j1[dx];
	  out[0] += du[2*j] * k;
	  out[1] += du[2*j+1] * k;
	}
      }
    }
  }
  target[0] = out[0];
  target[1] = out[1];
}

template<typename FloatType>
void spread_subproblem_1d(int64_t off1, int64_t size1,FloatType *du,int64_t M,
			  FloatType *kx,FloatType *dd, const SpreadParameters<FloatType>& opts)
/* 1D spreader from nonuniform to uniform subproblem grid, without wrapping.
   Inputs:
   off1 - integer offset of left end of du subgrid from that of overall fine
          periodized output grid {0,1,..N-1}.
   size1 - integer length of output subgrid du
   M - number of NU pts in subproblem
   kx (length M) - are rescaled NU source locations, should lie in
                   [off1+ns/2,off1+size1-1-ns/2] so as kernels stay in bounds
   dd (length M complex, interleaved) - source strengths
   Outputs:
   du (length size1 complex, interleaved) - preallocated uniform subgrid array

   The reason periodic wrapping is avoided in subproblems is speed: avoids
   conditionals, indirection (pointers), and integer mod. Originally 2017.
   Kernel eval mods by Ludvig al Klinteberg.
   Fixed so rounding to integer grid consistent w/ get_subgrid, prevents
   chance of segfault when epsmach*N1>O(1), assuming max() and ceil() commute.
   This needed off1 as extra arg. AHB 11/30/20.
*/
{
  int ns=opts.kernel_width;          // a.k.a. w
  FloatType ns2 = (FloatType)ns/2;          // half spread width
  for (int64_t i=0;i<2*size1;++i)         // zero output
    du[i] = 0.0;
  FloatType kernel_args[MAX_KERNEL_WIDTH];
  FloatType ker[MAX_KERNEL_WIDTH];
  for (int64_t i=0; i<M; i++) {           // loop over NU pts
    FloatType re0 = dd[2*i];
    FloatType im0 = dd[2*i+1];
    // ceil offset, hence rounding, must match that in get_subgrid...
    int64_t i1 = (int64_t)std::ceil(kx[i] - ns2);    // fine grid start index
    FloatType x1 = (FloatType)i1 - kx[i];            // x1 in [-w/2,-w/2+1], up to rounding
    // However if N1*epsmach>O(1) then can cause O(1) errors in x1, hence ppoly
    // kernel evaluation will fall outside their designed domains, >>1 errors.
    // This can only happen if the overall error would be O(1) anyway. Clip x1??
    if (x1<-ns2) x1=-ns2;
    if (x1>-ns2+1) x1=-ns2+1;   // ***
    if (opts.kerevalmeth==0) {          // faster Horner poly method
      set_kernel_args(kernel_args, x1, opts);
      evaluate_kernel_vector(ker, kernel_args, opts, ns);
    } else
      eval_kernel_vec_Horner(ker,x1,ns,opts);
    int64_t j = i1-off1;    // offset rel to subgrid, starts the output indices
    // critical inner loop:
    for (int dx=0; dx<ns; ++dx) {
      FloatType k = ker[dx];
      du[2*j] += re0*k;
      du[2*j+1] += im0*k;
      ++j;
    }
  }
}

template<typename FloatType>
void spread_subproblem_2d(int64_t off1,int64_t off2,int64_t size1,int64_t size2,
                          FloatType *du,int64_t M, FloatType *kx,FloatType *ky,FloatType *dd,
			  const SpreadParameters<FloatType>& opts)
/* spreader from dd (NU) to du (uniform) in 2D without wrapping.
   See above docs/notes for spread_subproblem_2d.
   kx,ky (size M) are NU locations in [off+ns/2,off+size-1-ns/2] in both dims.
   dd (size M complex) are complex source strengths
   du (size size1*size2) is complex uniform output array
 */
{
  int ns=opts.kernel_width;
  FloatType ns2 = (FloatType)ns/2;          // half spread width
  for (int64_t i=0;i<2*size1*size2;++i)
    du[i] = 0.0;
  FloatType kernel_args[2*MAX_KERNEL_WIDTH];
  // Kernel values stored in consecutive memory. This allows us to compute
  // values in two directions in a single kernel evaluation call.
  FloatType kernel_values[2*MAX_KERNEL_WIDTH];
  FloatType *ker1 = kernel_values;
  FloatType *ker2 = kernel_values + ns;
  for (int64_t i=0; i<M; i++) {           // loop over NU pts
    FloatType re0 = dd[2*i];
    FloatType im0 = dd[2*i+1];
    // ceil offset, hence rounding, must match that in get_subgrid...
    int64_t i1 = (int64_t)std::ceil(kx[i] - ns2);   // fine grid start indices
    int64_t i2 = (int64_t)std::ceil(ky[i] - ns2);
    FloatType x1 = (FloatType)i1 - kx[i];
    FloatType x2 = (FloatType)i2 - ky[i];
    if (opts.kerevalmeth==0) {          // faster Horner poly method
      set_kernel_args(kernel_args, x1, opts);
      set_kernel_args(kernel_args+ns, x2, opts);
      evaluate_kernel_vector(kernel_values, kernel_args, opts, 2*ns);
    } else {
      eval_kernel_vec_Horner(ker1,x1,ns,opts);
      eval_kernel_vec_Horner(ker2,x2,ns,opts);
    }
    // Combine kernel with complex source value to simplify inner loop
    FloatType ker1val[2*MAX_KERNEL_WIDTH];    // here 2* is because of complex
    for (int i = 0; i < ns; i++) {
      ker1val[2*i] = re0*ker1[i];
      ker1val[2*i+1] = im0*ker1[i];
    }
    // critical inner loop:
    for (int dy=0; dy<ns; ++dy) {
      int64_t j = size1*(i2-off2+dy) + i1-off1;   // should be in subgrid
      FloatType kerval = ker2[dy];
      FloatType *trg = du+2*j;
      for (int dx=0; dx<2*ns; ++dx) {
	trg[dx] += kerval*ker1val[dx];
      }
    }
  }
}

template<typename FloatType>
void spread_subproblem_3d(int64_t off1,int64_t off2,int64_t off3,int64_t size1,
                          int64_t size2,int64_t size3,FloatType *du,int64_t M,
			  FloatType *kx,FloatType *ky,FloatType *kz,FloatType *dd,
			  const SpreadParameters<FloatType>& opts)
/* spreader from dd (NU) to du (uniform) in 3D without wrapping.
   See above docs/notes for spread_subproblem_2d.
   kx,ky,kz (size M) are NU locations in [off+ns/2,off+size-1-ns/2] in each rank.
   dd (size M complex) are complex source strengths
   du (size size1*size2*size3) is uniform complex output array
 */
{
  int ns=opts.kernel_width;
  FloatType ns2 = (FloatType)ns/2;          // half spread width
  for (int64_t i=0;i<2*size1*size2*size3;++i)
    du[i] = 0.0;
  FloatType kernel_args[3*MAX_KERNEL_WIDTH];
  // Kernel values stored in consecutive memory. This allows us to compute
  // values in all three directions in a single kernel evaluation call.
  FloatType kernel_values[3*MAX_KERNEL_WIDTH];
  FloatType *ker1 = kernel_values;
  FloatType *ker2 = kernel_values + ns;
  FloatType *ker3 = kernel_values + 2*ns;
  for (int64_t i=0; i<M; i++) {           // loop over NU pts
    FloatType re0 = dd[2*i];
    FloatType im0 = dd[2*i+1];
    // ceil offset, hence rounding, must match that in get_subgrid...
    int64_t i1 = (int64_t)std::ceil(kx[i] - ns2);   // fine grid start indices
    int64_t i2 = (int64_t)std::ceil(ky[i] - ns2);
    int64_t i3 = (int64_t)std::ceil(kz[i] - ns2);
    FloatType x1 = (FloatType)i1 - kx[i];
    FloatType x2 = (FloatType)i2 - ky[i];
    FloatType x3 = (FloatType)i3 - kz[i];
    if (opts.kerevalmeth==0) {          // faster Horner poly method
      set_kernel_args(kernel_args, x1, opts);
      set_kernel_args(kernel_args+ns, x2, opts);
      set_kernel_args(kernel_args+2*ns, x3, opts);
      evaluate_kernel_vector(kernel_values, kernel_args, opts, 3*ns);
    } else {
      eval_kernel_vec_Horner(ker1,x1,ns,opts);
      eval_kernel_vec_Horner(ker2,x2,ns,opts);
      eval_kernel_vec_Horner(ker3,x3,ns,opts);
    }
    // Combine kernel with complex source value to simplify inner loop
    FloatType ker1val[2*MAX_KERNEL_WIDTH];    // here 2* is because of complex
    for (int i = 0; i < ns; i++) {
      ker1val[2*i] = re0*ker1[i];
      ker1val[2*i+1] = im0*ker1[i];
    }
    // critical inner loop:
    for (int dz=0; dz<ns; ++dz) {
      int64_t oz = size1*size2*(i3-off3+dz);        // offset due to z
      for (int dy=0; dy<ns; ++dy) {
	int64_t j = oz + size1*(i2-off2+dy) + i1-off1;   // should be in subgrid
	FloatType kerval = ker2[dy]*ker3[dz];
	FloatType *trg = du+2*j;
	for (int dx=0; dx<2*ns; ++dx) {
	  trg[dx] += kerval*ker1val[dx];
	}
      }
    }
  }
}

template<typename FloatType>
void add_wrapped_subgrid(int64_t offset1,int64_t offset2,int64_t offset3,
			 int64_t size1,int64_t size2,int64_t size3,int64_t N1,
			 int64_t N2,int64_t N3,FloatType *data_uniform, FloatType *du0)
/* Add a large subgrid (du0) to output grid (data_uniform),
   with periodic wrapping to N1,N2,N3 box.
   offset1,2,3 give the offset of the subgrid from the lowest corner of output.
   size1,2,3 give the size of subgrid.
   Works in all dims. Not thread-safe and must be called inside omp critical.
   Barnett 3/27/18 made separate routine, tried to speed up inner loop.
*/
{
  std::vector<int64_t> o2(size2), o3(size3);
  int64_t y=offset2, z=offset3;    // fill wrapped ptr lists in slower dims y,z...
  for (int i=0; i<size2; ++i) {
    if (y<0) y+=N2;
    if (y>=N2) y-=N2;
    o2[i] = y++;
  }
  for (int i=0; i<size3; ++i) {
    if (z<0) z+=N3;
    if (z>=N3) z-=N3;
    o3[i] = z++;
  }
  int64_t nlo = (offset1<0) ? -offset1 : 0;          // # wrapping below in x
  int64_t nhi = (offset1+size1>N1) ? offset1+size1-N1 : 0;    // " above in x
  // this triple loop works in all dims
  for (int dz=0; dz<size3; dz++) {       // use ptr lists in each axis
    int64_t oz = N1*N2*o3[dz];            // offset due to z (0 in <3D)
    for (int dy=0; dy<size2; dy++) {
      int64_t oy = oz + N1*o2[dy];        // off due to y & z (0 in 1D)
      FloatType *out = data_uniform + 2*oy;
      FloatType *in  = du0 + 2*size1*(dy + size2*dz);   // ptr to subgrid array
      int64_t o = 2*(offset1+N1);         // 1d offset for output
      for (int j=0; j<2*nlo; j++)        // j is really dx/2 (since re,im parts)
	out[j+o] += in[j];
      o = 2*offset1;
      for (int j=2*nlo; j<2*(size1-nhi); j++)
	out[j+o] += in[j];
      o = 2*(offset1-N1);
      for (int j=2*(size1-nhi); j<2*size1; j++)
      	out[j+o] += in[j];
    }
  }
}

template<typename FloatType>
void add_wrapped_subgrid_thread_safe(int64_t offset1,int64_t offset2,int64_t offset3,
                                     int64_t size1,int64_t size2,int64_t size3,int64_t N1,
                                     int64_t N2,int64_t N3,FloatType *data_uniform, FloatType *du0)
/* Add a large subgrid (du0) to output grid (data_uniform),
   with periodic wrapping to N1,N2,N3 box.
   offset1,2,3 give the offset of the subgrid from the lowest corner of output.
   size1,2,3 give the size of subgrid.
   Works in all dims. Thread-safe variant of the above routine,
   using atomic writes (R Blackwell, Nov 2020).
*/
{
  std::vector<int64_t> o2(size2), o3(size3);
  int64_t y=offset2, z=offset3;    // fill wrapped ptr lists in slower dims y,z...
  for (int i=0; i<size2; ++i) {
    if (y<0) y+=N2;
    if (y>=N2) y-=N2;
    o2[i] = y++;
  }
  for (int i=0; i<size3; ++i) {
    if (z<0) z+=N3;
    if (z>=N3) z-=N3;
    o3[i] = z++;
  }
  int64_t nlo = (offset1<0) ? -offset1 : 0;          // # wrapping below in x
  int64_t nhi = (offset1+size1>N1) ? offset1+size1-N1 : 0;    // " above in x
  // this triple loop works in all dims
  for (int dz=0; dz<size3; dz++) {       // use ptr lists in each axis
    int64_t oz = N1*N2*o3[dz];            // offset due to z (0 in <3D)
    for (int dy=0; dy<size2; dy++) {
      int64_t oy = oz + N1*o2[dy];        // off due to y & z (0 in 1D)
      FloatType *out = data_uniform + 2*oy;
      FloatType *in  = du0 + 2*size1*(dy + size2*dz);   // ptr to subgrid array
      int64_t o = 2*(offset1+N1);         // 1d offset for output
      for (int j=0; j<2*nlo; j++) { // j is really dx/2 (since re,im parts)
#pragma omp atomic
        out[j + o] += in[j];
      }
      o = 2*offset1;
      for (int j=2*nlo; j<2*(size1-nhi); j++) {
#pragma omp atomic
        out[j + o] += in[j];
      }
      o = 2*(offset1-N1);
      for (int j=2*(size1-nhi); j<2*size1; j++) {
#pragma omp atomic
        out[j+o] += in[j];
      }
    }
  }
}

template<typename FloatType>
void get_subgrid(int64_t &offset1,int64_t &offset2,int64_t &offset3,int64_t &size1,int64_t &size2,int64_t &size3,int64_t M,FloatType* kx,FloatType* ky,FloatType* kz,int ns,int ndims)
/* Writes out the integer offsets and sizes of a "subgrid" (cuboid subset of
   Z^ndims) large enough to enclose all of the nonuniform points with
   (non-periodic) padding of half the kernel width ns to each side in
   each relevant dimension.

 Inputs:
   M - number of nonuniform points, ie, length of kx array (and ky if ndims>1,
       and kz if ndims>2)
   kx,ky,kz - coords of nonuniform points (ky only read if ndims>1,
              kz only read if ndims>2). To be useful for spreading, they are
              assumed to be in [0,Nj] for dimension j=1,..,ndims.
   ns - (positive integer) spreading kernel width.
   ndims - space dimension (1,2, or 3).

 Outputs:
   offset1,2,3 - left-most coord of cuboid in each dimension (up to ndims)
   size1,2,3   - size of cuboid in each dimension.
                 Thus the right-most coord of cuboid is offset+size-1.
   Returns offset 0 and size 1 for each unused dimension (ie when ndims<3);
   this is required by the calling code.

 Example:
      inputs:
          ndims=1, M=2, kx[0]=0.2, ks[1]=4.9, ns=3
      outputs:
          offset1=-1 (since kx[0] spreads to {-1,0,1}, and -1 is the min)
          size1=8 (since kx[1] spreads to {4,5,6}, so subgrid is {-1,..,6}
                   hence 8 grid points).
 Notes:
   1) Works in all dims 1,2,3.
   2) Rounding of the kx (and ky, kz) to the grid is tricky and must match the
   rounding step used in spread_subproblem_{1,2,3}d. Namely, the ceil of
   (the NU pt coord minus ns/2) gives the left-most index, in each dimension.
   This being done consistently is crucial to prevent segfaults in subproblem
   spreading. This assumes that max() and ceil() commute in the floating pt
   implementation.
   Originally by J Magland, 2017. AHB realised the rounding issue in
   6/16/17, but only fixed a rounding bug causing segfault in (highly
   inaccurate) single-precision with N1>>1e7 on 11/30/20.
   3) Requires O(M) RAM reads to find the k array bnds. Almost negligible in
   tests.
*/
{
  FloatType ns2 = (FloatType)ns/2;
  FloatType min_kx,max_kx;   // 1st (x) dimension: get min/max of nonuniform points
  array_range(M,kx,&min_kx,&max_kx);
  offset1 = (int64_t)std::ceil(min_kx-ns2);   // min index touched by kernel
  size1 = (int64_t)std::ceil(max_kx-ns2) - offset1 + ns;  // int(ceil) first!
  if (ndims>1) {
    FloatType min_ky,max_ky;   // 2nd (y) dimension: get min/max of nonuniform points
    array_range(M,ky,&min_ky,&max_ky);
    offset2 = (int64_t)std::ceil(min_ky-ns2);
    size2 = (int64_t)std::ceil(max_ky-ns2) - offset2 + ns;
  } else {
    offset2 = 0;
    size2 = 1;
  }
  if (ndims>2) {
    FloatType min_kz,max_kz;   // 3rd (z) dimension: get min/max of nonuniform points
    array_range(M,kz,&min_kz,&max_kz);
    offset3 = (int64_t)std::ceil(min_kz-ns2);
    size3 = (int64_t)std::ceil(max_kz-ns2) - offset3 + ns;
  } else {
    offset3 = 0;
    size3 = 1;
  }
}

}  // namespace

// Explicit instatiations.
template class Plan<CPUDevice, float>;
template class Plan<CPUDevice, double>;

}  // namespace nufft
}  // namespace tensorflow
