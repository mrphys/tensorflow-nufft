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

#if GOOGLE_CUDA

#include "tensorflow_nufft/cc/kernels/nufft_plan.h"

#include "tensorflow/core/platform/stream_executor.h"

#include "tensorflow_nufft/third_party/cuda_samples/helper_cuda.h"
#include "tensorflow_nufft/cc/kernels/nufft_util.h"
#include "tensorflow_nufft/cc/kernels/omp_api.h"


namespace tensorflow {
namespace nufft {

namespace {

template<typename FloatType>
constexpr cufftType kCufftType = CUFFT_C2C;
template<>
constexpr cufftType kCufftType<float> = CUFFT_C2C;
template<>
constexpr cufftType kCufftType<double> = CUFFT_Z2Z;

template<typename FloatType>
Status setup_spreader(int rank, FloatType eps, double upsampling_factor,
                      KernelEvaluationMethod kernel_evaluation_method,
                      SpreadOptions<FloatType>& spopts);

template<typename FloatType>
Status setup_spreader_for_nufft(int rank, FloatType eps,
                                const Options& options,
                                SpreadOptions<FloatType> &spopts);

void set_bin_sizes(TransformType type, int rank, Options& options);

template<typename FloatType>
Status set_grid_size(int ms,
                     int bin_size,
                     const Options& options,
                     const SpreadOptions<FloatType>& spopts,
                     int* grid_size);

template<typename FloatType>
Status allocate_gpu_memory_2d(Plan<GPUDevice, FloatType>* d_plan);

template<typename FloatType>
Status allocate_gpu_memory_3d(Plan<GPUDevice, FloatType>* d_plan);

template<typename FloatType>
void free_gpu_memory(Plan<GPUDevice, FloatType>* d_plan);

} // namespace


template<typename FloatType>
Plan<GPUDevice, FloatType>::Plan(
    OpKernelContext* context,
    TransformType type,
    int rank,
    gtl::InlinedVector<int, 4> num_modes,
    FftDirection fft_direction,
    int num_transforms,
    FloatType tol,
    const Options& options) {

  OP_REQUIRES(context,
              type != TransformType::TYPE_3,
              errors::Unimplemented("type-3 transforms are not implemented"));
  OP_REQUIRES(context, rank >= 2 && rank <= 3,
              errors::InvalidArgument("rank must be 2 or 3"));
  OP_REQUIRES(context, num_transforms >= 1,
              errors::InvalidArgument("num_transforms must be >= 1"));
  OP_REQUIRES(context, rank == num_modes.size(),
              errors::InvalidArgument("num_modes must have size equal to rank"));

  // auto* stream = context->op_device_context()->stream();
  // OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

  // int device_id = stream->parent()->device_ordinal();

  GPUDevice device = context->eigen_device<GPUDevice>();

  // std::cout << "device_id = " << device_id << std::endl;
  // cudaSetDevice(0);

  
  // TODO: check options.
  //  - If mode_order == FFT, raise unimplemented error.
  //  - If check_bounds == true, raise unimplemented error.

        // Mult-GPU support: set the CUDA Device ID:
        int orig_gpu_device_id;
        cudaGetDevice(& orig_gpu_device_id);
        // if (spopts == NULL) {
        //     // options might not be supplied to this function => assume device
        //     // 0 by default
        //     cudaSetDevice(0);
        // } else {
        cudaSetDevice(options.gpu_device_id);
        // }


  // Initialize all values to 0. TODO: move to initialization list.
  this->M = 0;
  this->nf1 = 0;
  this->nf2 = 0;
  this->nf3 = 0;
  this->ms = 0;
  this->mt = 0;
  this->mu = 0;
  this->totalnumsubprob = 0;
  this->byte_now = 0;
  this->kx = nullptr;
  this->ky = nullptr;
  this->kz = nullptr;
  this->c = nullptr;
  this->fk = nullptr;
  this->idxnupts = nullptr;
  this->sortidx = nullptr;
  this->numsubprob = nullptr;
  this->binsize = nullptr;
  this->binstartpts = nullptr;
  this->subprob_to_bin = nullptr;
  this->subprobstartpts = nullptr;
  this->finegridsize = nullptr;
  this->fgstartpts = nullptr;
  this->numnupts = nullptr;
  this->subprob_to_nupts = nullptr;

  // Copy options.
  this->options_ = options;

  // Select kernel evaluation method.
  if (this->options_.kernel_evaluation_method == KernelEvaluationMethod::AUTO) {
    this->options_.kernel_evaluation_method = KernelEvaluationMethod::DIRECT;
  }

  // Select upsampling factor. Currently always defaults to 2.
  if (this->options_.upsampling_factor == 0.0) {
    this->options_.upsampling_factor = 2.0;
  }

  // Configure threading (irrelevant for GPU computation, but is used for some
  // CPU computations).
  if (this->options_.num_threads == 0) {
    this->options_.num_threads = OMP_GET_MAX_THREADS();
  }

  // Select spreading method.
  if (this->options_.gpu_spread_method == GpuSpreadMethod::AUTO) {
    if (rank == 2 && type == TransformType::TYPE_1)
      this->options_.gpu_spread_method = GpuSpreadMethod::SUBPROBLEM;
    else if (rank == 2 && type == TransformType::TYPE_2)
      this->options_.gpu_spread_method = GpuSpreadMethod::NUPTS_DRIVEN;
    else if (rank == 3 && type == TransformType::TYPE_1)
      this->options_.gpu_spread_method = GpuSpreadMethod::SUBPROBLEM;
    else if (rank == 3 && type == TransformType::TYPE_2)
      this->options_.gpu_spread_method = GpuSpreadMethod::NUPTS_DRIVEN;
  }

  // This must be set before calling setup_spreader_for_nufft.
  this->spopts.spread_interp_only = this->options_.spread_interp_only;

  // Setup spreading options.
  OP_REQUIRES_OK(context,
                 setup_spreader_for_nufft(
                    rank, tol, this->options_, this->spopts));

  this->rank_ = rank;
  this->ms = num_modes[0];
  if (rank > 1)
    this->mt = num_modes[1];
  if (rank > 2)
    this->mu = num_modes[2];

  // Set the bin sizes.
  set_bin_sizes(type, rank, this->options_);

  // Set the grid sizes.
  int nf1 = 1, nf2 = 1, nf3 = 1;
  OP_REQUIRES_OK(context,
                 set_grid_size(this->ms, this->options_.gpu_obin_size.x,
                               this->options_, this->spopts, &nf1));
  if (rank > 1) {
    OP_REQUIRES_OK(context,
                   set_grid_size(this->mt, this->options_.gpu_obin_size.y,
                                 this->options_, this->spopts, &nf2));
  }
  if (rank > 2) {
    OP_REQUIRES_OK(context,
                   set_grid_size(this->mu, this->options_.gpu_obin_size.z,
                                 this->options_, this->spopts, &nf3));
  }

  this->nf1 = nf1;
  this->nf2 = nf2;
  this->nf3 = nf3;
  this->fft_direction_ = fft_direction;
  this->num_transforms_ = num_transforms;
  this->type_ = type;

  // Select maximum batch size.
  if (this->options_.max_batch_size == 0)
    // Heuristic from test codes.
    this->options_.max_batch_size = min(num_transforms, 8); 

  if (this->type_ == TransformType::TYPE_1)
    this->spopts.spread_direction = SpreadDirection::SPREAD;
  if (this->type_ == TransformType::TYPE_2)
    this->spopts.spread_direction = SpreadDirection::INTERP;

  switch(this->rank_)
  {
    case 2: {
      OP_REQUIRES_OK(context, allocate_gpu_memory_2d(this));
      break;
    }
    case 3: {
      OP_REQUIRES_OK(context, allocate_gpu_memory_3d(this));
      break;
    }
  }
  
  // Perform some actions not needed in spread/interp only mode.
  if (!this->options_.spread_interp_only)
  {
    // Allocate fine grid and set convenience pointer.
    int num_grid_elements = this->nf1 * this->nf2 * this->nf3;
    OP_REQUIRES_OK(context, context->allocate_temp(
        DataTypeToEnum<std::complex<FloatType>>::value,
        TensorShape({num_grid_elements * this->options_.max_batch_size}),
        &this->fine_grid_));
    this->fine_grid_data_ = reinterpret_cast<DType*>(
        this->fine_grid_.flat<std::complex<FloatType>>().data());

    // For each dimension, calculate Fourier coefficients of the kernel for
    // deconvolution. The computation is performed on the CPU before
    // transferring the results the GPU.
    int grid_sizes[3] = {this->nf1, this->nf2, this->nf3};
    Tensor kernel_fseries_host[3];
    FloatType* kernel_fseries_host_data[3];
    for (int i = 0; i < this->rank_; i++) {

      // Number of Fourier coefficients.      
      int num_coeffs = grid_sizes[i] / 2 + 1;

      // Allocate host memory and calculate the Fourier series on the CPU.
      AllocatorAttributes attr;
      attr.set_on_host(true);
      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DataTypeToEnum<FloatType>::value,
                                  TensorShape({num_coeffs}),
                                  &kernel_fseries_host[i], attr));
      kernel_fseries_host_data[i] = reinterpret_cast<FloatType*>(
          kernel_fseries_host[i].flat<FloatType>().data());
      kernel_fseries_1d(grid_sizes[i], this->spopts,
                        kernel_fseries_host_data[i]);

      // Allocate device memory.
      OP_REQUIRES_OK(context, context->allocate_temp(
                                  DataTypeToEnum<FloatType>::value,
                                  TensorShape({num_coeffs}),
                                  &this->kernel_fseries_[i]));

      // Save convenience accessors.
      this->kernel_fseries_data_[i] = reinterpret_cast<FloatType*>(
          this->kernel_fseries_[i].flat<FloatType>().data());

      // Now copy memory to device.
      size_t num_bytes = sizeof(FloatType) * num_coeffs;
      device.memcpyHostToDevice(
          reinterpret_cast<void*>(this->kernel_fseries_data_[i]),
          reinterpret_cast<void*>(kernel_fseries_host_data[i]),
          num_bytes);
    }

    // Make the cuFFT plan.
    cufftHandle fftplan;
    switch(this->rank_)
    {
      case 2:
      {
        int n[] = {nf2, nf1};
        int inembed[] = {nf2, nf1};

        cufftResult result = cufftPlanMany(
            &fftplan, rank, n, inembed, 1, inembed[0] * inembed[1],
            inembed, 1, inembed[0] * inembed[1],
            kCufftType<FloatType>, this->options_.max_batch_size);

        OP_REQUIRES(context, result == CUFFT_SUCCESS,
                    errors::Internal(
                        "cufftPlanMany failed with code: ", result));
      }
      break;
      case 3:
      {
        int rank = 3;
        int n[] = {nf3, nf2, nf1};
        int inembed[] = {nf3, nf2, nf1};
        int istride = 1;
        cufftResult result = cufftPlanMany(
            &fftplan,rank,n,inembed,istride,inembed[0]*inembed[1]*
            inembed[2],inembed,istride,inembed[0]*inembed[1]*inembed[2],
            kCufftType<FloatType>, this->options_.max_batch_size);

        OP_REQUIRES(context, result == CUFFT_SUCCESS,
                    errors::Internal(
                        "cufftPlanMany failed with code: ", result));
      }
      break;
    }
    this->fftplan = fftplan;
  }

  // Multi-GPU support: reset the device ID
  cudaSetDevice(orig_gpu_device_id);
}

template<typename FloatType>
Plan<GPUDevice, FloatType>::~Plan() {

  // Mult-GPU support: set the CUDA Device ID:
  int orig_gpu_device_id;
  cudaGetDevice(& orig_gpu_device_id);
  cudaSetDevice(this->options_.gpu_device_id);

	if (this->fftplan)
		cufftDestroy(this->fftplan);

  free_gpu_memory(this);

        // Multi-GPU support: reset the device ID
        cudaSetDevice(orig_gpu_device_id);
}

namespace {

template<typename FloatType>
Status setup_spreader(int rank, FloatType eps, double upsampling_factor,
                      KernelEvaluationMethod kernel_evaluation_method,
                      SpreadOptions<FloatType>& spopts)
// Initializes spreader kernel parameters given desired NUFFT tol eps,
// upsampling factor (=sigma in paper, or R in Dutt-Rokhlin), and ker eval meth
// (etiher 0:exp(sqrt()), 1: Horner ppval).
// Also sets all default options in SpreadOptions<FloatType>. See cnufftspread.h for spopts.
// Must call before any kernel evals done.
// Returns: 0 success, 1, warning, >1 failure (see error codes in utils.h)
{
  if (upsampling_factor != 2.0) {
    if (kernel_evaluation_method == KernelEvaluationMethod::HORNER) {
      return errors::Internal(
          "Horner kernel evaluation only supports the standard "
          "upsampling factor of 2.0, but got ", upsampling_factor);
    }
    if (upsampling_factor <= 1.0) {
      return errors::Internal(
          "upsampling_factor must be > 1.0, but is ", upsampling_factor);
    }
  }
    
  // defaults... (user can change after this function called)
  spopts.spread_direction = SpreadDirection::SPREAD;
  spopts.pirange = 1;             // user also should always set this
  spopts.upsampling_factor = upsampling_factor;

  // as in FINUFFT v2.0, allow too-small-eps by truncating to eps_mach...
  if (eps < kEpsilon<FloatType>) {
    eps = kEpsilon<FloatType>;
  }

  // Set kernel width w (aka ns) and ES kernel beta parameter, in spopts...
  int ns = std::ceil(-log10(eps / (FloatType)10.0));   // 1 digit per power of ten
  if (upsampling_factor != 2.0)           // override ns for custom sigma
    ns = std::ceil(-log(eps) / (kPi<FloatType> * sqrt(1.0 - 1.0 / upsampling_factor)));  // formula, gamma=1
  ns = max(2, ns);               // we don't have ns=1 version yet
  if (ns > kMaxKernelWidth) {         // clip to match allocated arrays
    ns = kMaxKernelWidth;
  }
  spopts.nspread = ns;

  spopts.ES_halfwidth = (FloatType)ns / 2;   // constants to help ker eval (except Horner)
  spopts.ES_c = 4.0 / (FloatType)(ns * ns);

  FloatType beta_over_ns = 2.30;         // gives decent betas for default sigma=2.0
  if (ns == 2) beta_over_ns = 2.20;  // some small-width tweaks...
  if (ns == 3) beta_over_ns = 2.26;
  if (ns == 4) beta_over_ns = 2.38;
  if (upsampling_factor != 2.0) {          // again, override beta for custom sigma
    FloatType gamma=0.97;              // must match devel/gen_all_horner_C_code.m
    beta_over_ns = gamma * kPi<FloatType> * (1-1/(2*upsampling_factor));  // formula based on cutoff
  }
  spopts.ES_beta = beta_over_ns * (FloatType)ns;    // set the kernel beta parameter

  if (spopts.spread_interp_only)
    spopts.ES_scale = calculate_scale_factor(rank, spopts);

  return Status::OK();
}

template<typename FloatType>
Status setup_spreader_for_nufft(int rank, FloatType eps,
                                const Options& options,
                                SpreadOptions<FloatType>& spopts)
// Set up the spreader parameters given eps, and pass across various nufft
// options. Report status of setup_spreader.  Barnett 10/30/17
{
  TF_RETURN_IF_ERROR(setup_spreader(
      rank, eps, options.upsampling_factor,
      options.kernel_evaluation_method, spopts));

  spopts.pirange = 1;
  spopts.num_threads = options.num_threads;

  return Status::OK();
}

void set_bin_sizes(TransformType type, int rank, Options& options)
{
  switch(rank)
  {
    case 2:
    {
      options.gpu_bin_size.x = (options.gpu_bin_size.x == 0) ? 32 :
          options.gpu_bin_size.x;
      options.gpu_bin_size.y = (options.gpu_bin_size.y == 0) ? 32 :
          options.gpu_bin_size.y;
      options.gpu_bin_size.z = 1;
    }
    break;
    case 3:
    {
      switch(options.gpu_spread_method)
      {
        case GpuSpreadMethod::NUPTS_DRIVEN:
        case GpuSpreadMethod::SUBPROBLEM:
        {
          options.gpu_bin_size.x = (options.gpu_bin_size.x == 0) ? 16 :
              options.gpu_bin_size.x;
          options.gpu_bin_size.y = (options.gpu_bin_size.y == 0) ? 16 :
              options.gpu_bin_size.y;
          options.gpu_bin_size.z = (options.gpu_bin_size.z == 0) ? 2 :
              options.gpu_bin_size.z;
        }
        break;
        case GpuSpreadMethod::BLOCK_GATHER:
        {
          options.gpu_obin_size.x = (options.gpu_obin_size.x == 0) ? 8 :
              options.gpu_obin_size.x;
          options.gpu_obin_size.y = (options.gpu_obin_size.y == 0) ? 8 :
              options.gpu_obin_size.y;
          options.gpu_obin_size.z = (options.gpu_obin_size.z == 0) ? 8 :
              options.gpu_obin_size.z;
          options.gpu_bin_size.x = (options.gpu_bin_size.x == 0) ? 4 :
              options.gpu_bin_size.x;
          options.gpu_bin_size.y = (options.gpu_bin_size.y == 0) ? 4 :
              options.gpu_bin_size.y;
          options.gpu_bin_size.z = (options.gpu_bin_size.z == 0) ? 4 :
              options.gpu_bin_size.z;
        }
        break;
      }
    }
    break;
  }
}

template<typename FloatType>
Status set_grid_size(int ms,
                     int bin_size,
                     const Options& options,
                     const SpreadOptions<FloatType>& spopts,
                     int* grid_size) {
  // for spread/interp only, we do not apply oversampling (Montalt 6/8/2021).
  if (options.spread_interp_only) {
    *grid_size = ms;
  } else {
    *grid_size = static_cast<int>(options.upsampling_factor * ms);
  }

  // This is required to avoid errors.
  if (*grid_size < 2 * spopts.nspread)
    *grid_size = 2 * spopts.nspread;

  // Check if array size is too big.
  if (*grid_size > kMaxArraySize) {
    return errors::Internal(
        "Upsampled dim size too big: ", *grid_size, " > ", kMaxArraySize);
  }

  // Find the next smooth integer.
  if (options.gpu_spread_method == GpuSpreadMethod::BLOCK_GATHER)
    *grid_size = next_smooth_int(*grid_size, bin_size);
  else
    *grid_size = next_smooth_int(*grid_size);

  // For spread/interp only mode, make sure that the grid size is valid.
  if (options.spread_interp_only && *grid_size != ms) {
    return errors::Internal(
        "Invalid grid size: ", ms, ". Value should be even, "
        "larger than the kernel (", 2 * spopts.nspread, ") and have no prime "
        "factors larger than 5.");
  }

  return Status::OK();
}

template<typename FloatType>
Status allocate_gpu_memory_2d(Plan<GPUDevice, FloatType>* d_plan) {

  // Mult-GPU support: set the CUDA Device ID:
  int orig_gpu_device_id;
  cudaGetDevice(& orig_gpu_device_id);
  cudaSetDevice(d_plan->options_.gpu_device_id);

  int nf1 = d_plan->nf1;
  int nf2 = d_plan->nf2;

  d_plan->byte_now=0;
  // No extra memory is needed in nuptsdriven method (case 1)
  switch (d_plan->options_.gpu_spread_method)
  {
    case GpuSpreadMethod::NUPTS_DRIVEN:
      {
        if (d_plan->options_.gpu_sort_points) {
          int numbins[2];
          numbins[0] = ceil((FloatType) nf1/d_plan->options_.gpu_bin_size.x);
          numbins[1] = ceil((FloatType) nf2/d_plan->options_.gpu_bin_size.y);
          checkCudaErrors(cudaMalloc(&d_plan->binsize,numbins[0]*
            numbins[1]*sizeof(int)));
          checkCudaErrors(cudaMalloc(&d_plan->binstartpts,numbins[0]*
            numbins[1]*sizeof(int)));
        }
      }
      break;
    case GpuSpreadMethod::SUBPROBLEM:
      {
        int numbins[2];
        numbins[0] = ceil((FloatType) nf1/d_plan->options_.gpu_bin_size.x);
        numbins[1] = ceil((FloatType) nf2/d_plan->options_.gpu_bin_size.y);
        checkCudaErrors(cudaMalloc(&d_plan->numsubprob,numbins[0]*
            numbins[1]*sizeof(int)));
        checkCudaErrors(cudaMalloc(&d_plan->binsize,numbins[0]*
            numbins[1]*sizeof(int)));
        checkCudaErrors(cudaMalloc(&d_plan->binstartpts,numbins[0]*
            numbins[1]*sizeof(int)));
        checkCudaErrors(cudaMalloc(&d_plan->subprobstartpts,
            (numbins[0]*numbins[1]+1)*sizeof(int)));
      }
      break;
    case GpuSpreadMethod::PAUL:
      {
        int numbins[2];
        numbins[0] = ceil((FloatType) nf1/d_plan->options_.gpu_bin_size.x);
        numbins[1] = ceil((FloatType) nf2/d_plan->options_.gpu_bin_size.y);
        checkCudaErrors(cudaMalloc(&d_plan->finegridsize,nf1*nf2*
            sizeof(int)));
        checkCudaErrors(cudaMalloc(&d_plan->fgstartpts,nf1*nf2*
            sizeof(int)));
        checkCudaErrors(cudaMalloc(&d_plan->numsubprob,numbins[0]*
            numbins[1]*sizeof(int)));
        checkCudaErrors(cudaMalloc(&d_plan->binsize,numbins[0]*
            numbins[1]*sizeof(int)));
        checkCudaErrors(cudaMalloc(&d_plan->binstartpts,numbins[0]*
            numbins[1]*sizeof(int)));
        checkCudaErrors(cudaMalloc(&d_plan->subprobstartpts,
            (numbins[0]*numbins[1]+1)*sizeof(int)));
      }
      break;
    default:
      return errors::Internal("Invalid GPU spread method");
  }

  // Multi-GPU support: reset the device ID
  cudaSetDevice(orig_gpu_device_id);
  
  return Status::OK();
}

template<typename FloatType>
Status allocate_gpu_memory_3d(Plan<GPUDevice, FloatType>* d_plan) {

        // Mult-GPU support: set the CUDA Device ID:
        int orig_gpu_device_id;
        cudaGetDevice(& orig_gpu_device_id);
        cudaSetDevice(d_plan->options_.gpu_device_id);

  int nf1 = d_plan->nf1;
  int nf2 = d_plan->nf2;
  int nf3 = d_plan->nf3;

  d_plan->byte_now=0;

  switch(d_plan->options_.gpu_spread_method)
  {
    case GpuSpreadMethod::NUPTS_DRIVEN:
      {
        if (d_plan->options_.gpu_sort_points) {
          int numbins[3];
          numbins[0] = ceil((FloatType) nf1/d_plan->options_.gpu_bin_size.x);
          numbins[1] = ceil((FloatType) nf2/d_plan->options_.gpu_bin_size.y);
          numbins[2] = ceil((FloatType) nf3/d_plan->options_.gpu_bin_size.z);
          checkCudaErrors(cudaMalloc(&d_plan->binsize,numbins[0]*
            numbins[1]*numbins[2]*sizeof(int)));
          checkCudaErrors(cudaMalloc(&d_plan->binstartpts,numbins[0]*
            numbins[1]*numbins[2]*sizeof(int)));
        }
      }
      break;
    case GpuSpreadMethod::SUBPROBLEM:
      {
        int numbins[3];
        numbins[0] = ceil((FloatType) nf1/d_plan->options_.gpu_bin_size.x);
        numbins[1] = ceil((FloatType) nf2/d_plan->options_.gpu_bin_size.y);
        numbins[2] = ceil((FloatType) nf3/d_plan->options_.gpu_bin_size.z);
        checkCudaErrors(cudaMalloc(&d_plan->numsubprob,numbins[0]*
          numbins[1]*numbins[2]*sizeof(int)));
        checkCudaErrors(cudaMalloc(&d_plan->binsize,numbins[0]*
          numbins[1]*numbins[2]*sizeof(int)));
        checkCudaErrors(cudaMalloc(&d_plan->binstartpts,numbins[0]*
          numbins[1]*numbins[2]*sizeof(int)));
        checkCudaErrors(cudaMalloc(&d_plan->subprobstartpts,
          (numbins[0]*numbins[1]*numbins[2]+1)*sizeof(int)));
      }
      break;
    case GpuSpreadMethod::BLOCK_GATHER:
      {
        int numobins[3], numbins[3];
        int binsperobins[3];
        numobins[0] = ceil((FloatType) nf1/d_plan->options_.gpu_obin_size.x);
        numobins[1] = ceil((FloatType) nf2/d_plan->options_.gpu_obin_size.y);
        numobins[2] = ceil((FloatType) nf3/d_plan->options_.gpu_obin_size.z);

        binsperobins[0] = d_plan->options_.gpu_obin_size.x/
          d_plan->options_.gpu_bin_size.x;
        binsperobins[1] = d_plan->options_.gpu_obin_size.y/
          d_plan->options_.gpu_bin_size.y;
        binsperobins[2] = d_plan->options_.gpu_obin_size.z/
          d_plan->options_.gpu_bin_size.z;

        numbins[0] = numobins[0]*(binsperobins[0]+2);
        numbins[1] = numobins[1]*(binsperobins[1]+2);
        numbins[2] = numobins[2]*(binsperobins[2]+2);

        checkCudaErrors(cudaMalloc(&d_plan->numsubprob,
          numobins[0]*numobins[1]*numobins[2]*sizeof(int)));
        checkCudaErrors(cudaMalloc(&d_plan->binsize,
          numbins[0]*numbins[1]*numbins[2]*sizeof(int)));
        checkCudaErrors(cudaMalloc(&d_plan->binstartpts,
          (numbins[0]*numbins[1]*numbins[2]+1)*sizeof(int)));
        checkCudaErrors(cudaMalloc(&d_plan->subprobstartpts,(numobins[0]
          *numobins[1]*numobins[2]+1)*sizeof(int)));
      }
      break;
    default:
      return errors::Internal("Invalid GPU spread method");
  }

  // Multi-GPU support: reset the device ID
  cudaSetDevice(orig_gpu_device_id);

  return Status::OK();
}

template<typename FloatType>
void free_gpu_memory(Plan<GPUDevice, FloatType>* d_plan) {

      // Mult-GPU support: set the CUDA Device ID:
      int orig_gpu_device_id;
      cudaGetDevice(& orig_gpu_device_id);
      cudaSetDevice(d_plan->options_.gpu_device_id);

	switch(d_plan->options_.gpu_spread_method)
	{
		case GpuSpreadMethod::NUPTS_DRIVEN:
			{
				if (d_plan->options_.gpu_sort_points) {
          if (d_plan->idxnupts)
					  checkCudaErrors(cudaFree(d_plan->idxnupts));
          if (d_plan->sortidx)
					  checkCudaErrors(cudaFree(d_plan->sortidx));
					checkCudaErrors(cudaFree(d_plan->binsize));
					checkCudaErrors(cudaFree(d_plan->binstartpts));
				}else{
          if (d_plan->idxnupts)
					  checkCudaErrors(cudaFree(d_plan->idxnupts));
				}
			}
			break;
		case GpuSpreadMethod::SUBPROBLEM:
			{
        if (d_plan->idxnupts)
				  checkCudaErrors(cudaFree(d_plan->idxnupts));
        if (d_plan->sortidx)
				  checkCudaErrors(cudaFree(d_plan->sortidx));
				checkCudaErrors(cudaFree(d_plan->numsubprob));
				checkCudaErrors(cudaFree(d_plan->binsize));
				checkCudaErrors(cudaFree(d_plan->binstartpts));
				checkCudaErrors(cudaFree(d_plan->subprobstartpts));
				checkCudaErrors(cudaFree(d_plan->subprob_to_bin));
			}
			break;
		case GpuSpreadMethod::PAUL:
			{
        if (d_plan->idxnupts)
				  checkCudaErrors(cudaFree(d_plan->idxnupts));
        if (d_plan->sortidx)
				  checkCudaErrors(cudaFree(d_plan->sortidx));
				checkCudaErrors(cudaFree(d_plan->numsubprob));
				checkCudaErrors(cudaFree(d_plan->binsize));
				checkCudaErrors(cudaFree(d_plan->finegridsize));
				checkCudaErrors(cudaFree(d_plan->binstartpts));
				checkCudaErrors(cudaFree(d_plan->subprobstartpts));
				checkCudaErrors(cudaFree(d_plan->subprob_to_bin));
			}
			break;
    case GpuSpreadMethod::BLOCK_GATHER:
			{
        if (d_plan->idxnupts)
				  checkCudaErrors(cudaFree(d_plan->idxnupts));
        if (d_plan->sortidx)
				  checkCudaErrors(cudaFree(d_plan->sortidx));
				checkCudaErrors(cudaFree(d_plan->numsubprob));
				checkCudaErrors(cudaFree(d_plan->binsize));
				checkCudaErrors(cudaFree(d_plan->binstartpts));
				checkCudaErrors(cudaFree(d_plan->subprobstartpts));
				checkCudaErrors(cudaFree(d_plan->subprob_to_bin));
			}
			break;
	}

        // Multi-GPU support: reset the device ID
        cudaSetDevice(orig_gpu_device_id);
}

} // namespace

template class Plan<GPUDevice, float>;
template class Plan<GPUDevice, double>;

} // namespace nufft
} // namespace tensorflow

#endif // GOOGLE_CUDA
