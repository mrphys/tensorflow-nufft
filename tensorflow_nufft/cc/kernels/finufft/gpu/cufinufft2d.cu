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

#include <iostream>
#include <iomanip>
#include <math.h>
#include <tensorflow_nufft/third_party/cuda_samples/helper_cuda.h>
#include <complex>
#include <cufft.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include "tensorflow_nufft/cc/kernels/finufft/gpu/cufinufft_eitherprec.h"
#include "tensorflow_nufft/cc/kernels/finufft/gpu/cuspreadinterp.h"
#include "tensorflow_nufft/cc/kernels/finufft/gpu/cudeconvolve.h"

using namespace std;
using namespace tensorflow;
using namespace tensorflow::nufft;


int CUFINUFFT2D1_EXEC(CUCPX* d_c, CUCPX* d_fk, Plan<GPUDevice, FLT>* d_plan)
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
{
  int blksize;
  int ier;
  CUCPX* d_fkstart;
  CUCPX* d_cstart;
  for (int i=0; i*d_plan->options_.max_batch_size < d_plan->num_transforms_; i++) {
    blksize = min(d_plan->num_transforms_ - i*d_plan->options_.max_batch_size, 
      d_plan->options_.max_batch_size);
    d_cstart   = d_c + i*d_plan->options_.max_batch_size*d_plan->num_points_;
    d_fkstart  = d_fk + i*d_plan->options_.max_batch_size*d_plan->mode_count_;
    d_plan->c  = d_cstart;
    d_plan->fk = d_fkstart;

    checkCudaErrors(cudaMemset(d_plan->fine_grid_data_,0,d_plan->options_.max_batch_size*
        d_plan->grid_size_ * sizeof(CUCPX)));

    // Step 1: Spread
    ier = CUSPREAD2D(d_plan,blksize);
    if (ier != 0 ) {
      printf("error: cuspread2d, method(%d)\n", d_plan->options_.spread_method);
      return ier;
    }

    // Step 2: FFT
    cufftResult result = CUFFT_EX(
      d_plan->fft_plan_, d_plan->fine_grid_data_, d_plan->fine_grid_data_,
      static_cast<int>(d_plan->fft_direction_));
    if (result != CUFFT_SUCCESS) {
      fprintf(stderr,"[%s] CUFFT_EX failed with error code: %d\n",__func__,result);
        return ERR_CUFFT;
    }

    // Step 3: deconvolve and shuffle
    CUDECONVOLVE2D(d_plan, blksize);
  }
  return ier;
}

int CUFINUFFT2D2_EXEC(CUCPX* d_c, CUCPX* d_fk, Plan<GPUDevice, FLT>* d_plan)
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
{
  int blksize;
  int ier;
  CUCPX* d_fkstart;
  CUCPX* d_cstart;
  for (int i=0; i*d_plan->options_.max_batch_size < d_plan->num_transforms_; i++) {
    blksize = min(d_plan->num_transforms_ - i*d_plan->options_.max_batch_size, 
      d_plan->options_.max_batch_size);
    d_cstart  = d_c  + i*d_plan->options_.max_batch_size*d_plan->num_points_;
    d_fkstart = d_fk + i*d_plan->options_.max_batch_size*d_plan->mode_count_;

    d_plan->c = d_cstart;
    d_plan->fk = d_fkstart;

    // Step 1: amplify Fourier coeffs fk and copy into upsampled array fw
    CUDECONVOLVE2D(d_plan,blksize);

    // Step 2: FFT
    cudaDeviceSynchronize();
    cufftResult result = CUFFT_EX(
      d_plan->fft_plan_, d_plan->fine_grid_data_, d_plan->fine_grid_data_,
      static_cast<int>(d_plan->fft_direction_));
    if (result != CUFFT_SUCCESS) {
      fprintf(stderr,"[%s] CUFFT_EX failed with error code: %d\n",__func__,result);
        return ERR_CUFFT;
    }

    // Step 3: deconvolve and shuffle
    ier = CUINTERP2D(d_plan, blksize);
    if (ier != 0 ) {
      printf("error: cuinterp2d, method(%d)\n", d_plan->options_.spread_method);
      return ier;
    }
  }
  return ier;
}

int CUFINUFFT2D_INTERP(CUCPX* d_c, CUCPX* d_fk, Plan<GPUDevice, FLT>* d_plan)
{
  assert(d_plan->spread_params_.spread_direction == SpreadDirection::INTERP);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  int blksize;
  int ier;
  int gridsize = d_plan->ms*d_plan->mt;
  CUCPX* d_fkstart;
  CUCPX* d_cstart;
  
  for (int i=0; i*d_plan->options_.max_batch_size < d_plan->num_transforms_; i++) {
    blksize = min(d_plan->num_transforms_ - i*d_plan->options_.max_batch_size, 
      d_plan->options_.max_batch_size);
    d_cstart  = d_c  + i*d_plan->options_.max_batch_size*d_plan->num_points_;
    d_fkstart = d_fk + i*d_plan->options_.max_batch_size*gridsize;

    d_plan->c = d_cstart;
    d_plan->fine_grid_data_ = d_fkstart;

    cudaEventRecord(start);
    ier = CUINTERP2D(d_plan, blksize);
    if (ier != 0 ) {
      printf("error: cuinterp2d, method(%d)\n", d_plan->options_.spread_method);
      return ier;
    }
  }
  
  using namespace thrust::placeholders;
  thrust::device_ptr<FLT> dev_ptr((FLT*) d_c);
  thrust::transform(dev_ptr, dev_ptr + 2*d_plan->num_transforms_*d_plan->num_points_,
            dev_ptr, _1 * (FLT) d_plan->spread_params_.ES_scale); 
  
  return ier;
}

int CUFINUFFT2D_SPREAD(CUCPX* d_c, CUCPX* d_fk, Plan<GPUDevice, FLT>* d_plan)
{
  assert(d_plan->spread_params_.spread_direction == SpreadDirection::SPREAD);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  int blksize;
  int ier;
  int gridsize = d_plan->ms*d_plan->mt;
  CUCPX* d_fkstart;
  CUCPX* d_cstart;

  for (int i=0; i*d_plan->options_.max_batch_size < d_plan->num_transforms_; i++) {
    blksize = min(d_plan->num_transforms_ - i*d_plan->options_.max_batch_size, 
      d_plan->options_.max_batch_size);
    d_cstart   = d_c + i*d_plan->options_.max_batch_size*d_plan->num_points_;
    d_fkstart  = d_fk + i*d_plan->options_.max_batch_size*gridsize;
    
    d_plan->c  = d_cstart;
    d_plan->fine_grid_data_ = d_fkstart;

    cudaEventRecord(start);
    ier = CUSPREAD2D(d_plan,blksize);
    if (ier != 0 ) {
      printf("error: cuspread2d, method(%d)\n", d_plan->options_.spread_method);
      return ier;
    }
  }

  using namespace thrust::placeholders;
  thrust::device_ptr<FLT> dev_ptr((FLT*) d_fk);
  thrust::transform(dev_ptr, dev_ptr + 2*d_plan->num_transforms_*gridsize,
            dev_ptr, _1 * (FLT) d_plan->spread_params_.ES_scale); 

  return ier;
}
