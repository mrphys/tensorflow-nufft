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

#define EIGEN_USE_GPU

#include <tensorflow_nufft/third_party/cuda_samples/helper_cuda.h>
#include <iostream>
#include <iomanip>
#include <assert.h>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cuComplex.h>
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow_nufft/cc/kernels/finufft/gpu/cuspreadinterp.h"
#include "tensorflow_nufft/cc/kernels/finufft/gpu/memtransfer.h"

using namespace std;
using namespace tensorflow;
using namespace tensorflow::nufft;


__global__ void CalcBinSizeNoGhost2DKernel(int M, int nf1, int nf2, int  bin_size_x, 
    int bin_size_y, int nbinx, int nbiny, int* bin_size, FLT *x, FLT *y, 
    int* sortidx, int pirange) {
	int binidx, binx, biny;
	int oldidx;
	FLT x_rescaled,y_rescaled;
	for (int i=threadIdx.x+blockIdx.x*blockDim.x; i<M; i+=gridDim.x*blockDim.x) {
		x_rescaled=RESCALE(x[i], nf1, pirange);
		y_rescaled=RESCALE(y[i], nf2, pirange);
		binx = floor(x_rescaled/bin_size_x);
		binx = binx >= nbinx ? binx-1 : binx;
		binx = binx < 0 ? 0 : binx;
		biny = floor(y_rescaled/bin_size_y);
		biny = biny >= nbiny ? biny-1 : biny;
		biny = biny < 0 ? 0 : biny;
		binidx = binx+biny*nbinx;
		oldidx = atomicAdd(&bin_size[binidx], 1);
		sortidx[i] = oldidx;
		if (binx >= nbinx || biny >= nbiny) {
			sortidx[i] = -biny;
		}
	}
}

__global__ void CalcBinSizeNoGhost3DKernel(int M, int nf1, int nf2, int nf3,
    int bin_size_x, int bin_size_y, int bin_size_z,
    int nbinx, int nbiny, int nbinz, int* bin_size, FLT *x, FLT *y, FLT *z,
    int* sortidx, int pirange) {
	int binidx, binx, biny, binz;
	int oldidx;
	FLT x_rescaled,y_rescaled,z_rescaled;
	for (int i=threadIdx.x+blockIdx.x*blockDim.x; i<M; i+=gridDim.x*blockDim.x) {
		x_rescaled=RESCALE(x[i], nf1, pirange);
		y_rescaled=RESCALE(y[i], nf2, pirange);
		z_rescaled=RESCALE(z[i], nf3, pirange);
		binx = floor(x_rescaled/bin_size_x);
		binx = binx >= nbinx ? binx-1 : binx;
		binx = binx < 0 ? 0 : binx;

		biny = floor(y_rescaled/bin_size_y);
		biny = biny >= nbiny ? biny-1 : biny;
		biny = biny < 0 ? 0 : biny;

		binz = floor(z_rescaled/bin_size_z);
		binz = binz >= nbinz ? binz-1 : binz;
		binz = binz < 0 ? 0 : binz;
		binidx = binx+biny*nbinx+binz*nbinx*nbiny;
		oldidx = atomicAdd(&bin_size[binidx], 1);
		sortidx[i] = oldidx;
	}
}

__global__ void CalcInvertofGlobalSortIdx2DKernel(int M, int bin_size_x, int bin_size_y, 
    int nbinx,int nbiny, int* bin_startpts, int* sortidx, FLT *x, FLT *y, 
    int* index, int pirange, int nf1, int nf2) {
	int binx, biny;
	int binidx;
	FLT x_rescaled, y_rescaled;
	for (int i=threadIdx.x+blockIdx.x*blockDim.x; i<M; i+=gridDim.x*blockDim.x) {
		x_rescaled=RESCALE(x[i], nf1, pirange);
		y_rescaled=RESCALE(y[i], nf2, pirange);
		binx = floor(x_rescaled/bin_size_x);
		binx = binx >= nbinx ? binx-1 : binx;
		binx = binx < 0 ? 0 : binx;
		biny = floor(y_rescaled/bin_size_y);
		biny = biny >= nbiny ? biny-1 : biny;
		biny = biny < 0 ? 0 : biny;
		binidx = binx+biny*nbinx;

		index[bin_startpts[binidx]+sortidx[i]] = i;
	}
}

__global__ void CalcInvertofGlobalSortIdx3DKernel(int M, int bin_size_x, int bin_size_y,
    int bin_size_z, int nbinx, int nbiny, int nbinz, int* bin_startpts,
    int* sortidx, FLT *x, FLT *y, FLT *z, int* index, int pirange, int nf1,
    int nf2, int nf3) {
	int binx,biny,binz;
	int binidx;
	FLT x_rescaled,y_rescaled,z_rescaled;
	for (int i=threadIdx.x+blockIdx.x*blockDim.x; i<M; i+=gridDim.x*blockDim.x) {
		x_rescaled=RESCALE(x[i], nf1, pirange);
		y_rescaled=RESCALE(y[i], nf2, pirange);
		z_rescaled=RESCALE(z[i], nf3, pirange);
		binx = floor(x_rescaled/bin_size_x);
		binx = binx >= nbinx ? binx-1 : binx;
		binx = binx < 0 ? 0 : binx;
		biny = floor(y_rescaled/bin_size_y);
		biny = biny >= nbiny ? biny-1 : biny;
		biny = biny < 0 ? 0 : biny;
		binz = floor(z_rescaled/bin_size_z);
		binz = binz >= nbinz ? binz-1 : binz;
		binz = binz < 0 ? 0 : binz;
		binidx = CalcGlobalIdx_V2(binx,biny,binz,nbinx,nbiny,nbinz);

		index[bin_startpts[binidx]+sortidx[i]] = i;
	}
}

#ifdef SINGLE
#define TrivialGlobalSortIdxKernel TrivialGlobalSortIdxKernel_S
#else
#define TrivialGlobalSortIdxKernel TrivialGlobalSortIdxKernel_D
#endif
__global__ void TrivialGlobalSortIdxKernel(int M, int* index) {
	for (int i=threadIdx.x+blockIdx.x*blockDim.x; i<M; i+=gridDim.x*blockDim.x) {
		index[i] = i;
	}
}


int CUSPREAD2D(Plan<GPUDevice, FLT>* d_plan, int blksize)
/*
  A wrapper for different spreading methods.

  Methods available:
  (1) Non-uniform points driven
  (2) Subproblem
  (3) Paul

  Melody Shih 07/25/19
*/
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int ier;
  switch(d_plan->options_.spread_method)
  {
    case SpreadMethod::NUPTS_DRIVEN:
      {
        cudaEventRecord(start);
        ier = CUSPREAD2D_NUPTSDRIVEN(d_plan, blksize);
        if (ier != 0 ) {
          cout<<"error: cnufftspread2d_gpu_nuptsdriven"<<endl;
          return 1;
        }
      }
      break;
    case SpreadMethod::SUBPROBLEM:
      {
        cudaEventRecord(start);
        ier = CUSPREAD2D_SUBPROB(d_plan, blksize);
        if (ier != 0 ) {
          cout<<"error: cnufftspread2d_gpu_subprob"<<endl;
          return 1;
        }
      }
      break;
    case SpreadMethod::PAUL:
      {
        cudaEventRecord(start);
        ier = CUSPREAD2D_PAUL(d_plan, blksize);
        if (ier != 0 ) {
          cout<<"error: cnufftspread2d_gpu_paul"<<endl;
          return 1;
        }
      }
      break;
    default:
      cout<<"error: incorrect method, should be 1,2,3"<<endl;
      return 2;
  }
#ifdef SPREADTIME
  float milliseconds = 0;
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cout<<"[time  ]"<< " Spread " << milliseconds <<" ms"<<endl;
#endif
  return ier;
}

int CUSPREAD2D_NUPTSDRIVEN_PROP(Plan<GPUDevice, FLT>* d_plan) {
  
  int num_blocks = (d_plan->num_points_ + 1024 - 1) / 1024;
  int threads_per_block = 1024;

  if (d_plan->spread_params_.sort_points == SortPoints::YES) {
    int bin_size[3];
    bin_size[0] = d_plan->options_.gpu_bin_size.x;
    bin_size[1] = d_plan->options_.gpu_bin_size.y;
    bin_size[2] = d_plan->options_.gpu_bin_size.z;
    if (bin_size[0] < 0 || bin_size[1] < 0 || bin_size[2] < 0) {
      cout << "error: invalid binsize (binsizex, binsizey) = (";
      cout << bin_size[0] << "," << bin_size[1] << ")" << endl;
      return 1;
    }

    int num_bins[3] = {1, 1, 1};
    int bin_count = 1;
    for (int i = 0; i < d_plan->rank_; i++) {
      num_bins[i] = (d_plan->grid_dims_[i] + bin_size[i] - 1) / bin_size[i];
      bin_count *= num_bins[i];
    }

    // This may not be necessary.
    d_plan->device_.synchronize();

    // Calculate bin sizes.
    d_plan->device_.memset(d_plan->binsize, 0, bin_count * sizeof(int));
    switch (d_plan->rank_) {
      case 2:
        TF_CHECK_OK(GpuLaunchKernel(
            CalcBinSizeNoGhost2DKernel,
            num_blocks, threads_per_block, 0, d_plan->device_.stream(),
            d_plan->num_points_, d_plan->grid_dims_[0], d_plan->grid_dims_[1],
            bin_size[0], bin_size[1], num_bins[0], num_bins[1],
            d_plan->binsize, d_plan->points_[0], d_plan->points_[1], d_plan->sortidx,
            d_plan->spread_params_.pirange));
        break;
      case 3:
        CalcBinSizeNoGhost3DKernel<<<num_blocks, threads_per_block>>>(
          d_plan->num_points_, d_plan->grid_dims_[0], d_plan->grid_dims_[1],
          d_plan->grid_dims_[2],
          bin_size[0],bin_size[1],bin_size[2],num_bins[0],num_bins[1],num_bins[2],
          d_plan->binsize,d_plan->points_[0],d_plan->points_[1],d_plan->points_[2],
          d_plan->sortidx,d_plan->spread_params_.pirange);
        break;
    }

    thrust::device_ptr<int> d_bin_sizes(d_plan->binsize);
    thrust::device_ptr<int> d_bin_start_points(d_plan->binstartpts);
    thrust::exclusive_scan(d_bin_sizes, d_bin_sizes + bin_count,
                           d_bin_start_points);

    switch (d_plan->rank_) {
      case 2:
        TF_CHECK_OK(GpuLaunchKernel(
            CalcInvertofGlobalSortIdx2DKernel,
            num_blocks, threads_per_block, 0, d_plan->device_.stream(),
            d_plan->num_points_, bin_size[0], bin_size[1], num_bins[0],
            num_bins[1], d_plan->binstartpts, d_plan->sortidx,
            d_plan->points_[0], d_plan->points_[1], d_plan->idxnupts,
            d_plan->spread_params_.pirange, d_plan->grid_dims_[0],
            d_plan->grid_dims_[1]));
        break;
      case 3:
        CalcInvertofGlobalSortIdx3DKernel<<<num_blocks, threads_per_block>>>(
          d_plan->num_points_,bin_size[0],
          bin_size[1],bin_size[2],num_bins[0],num_bins[1],num_bins[2],
          d_plan->binstartpts,
          d_plan->sortidx,d_plan->points_[0],d_plan->points_[1],d_plan->points_[2],
          d_plan->idxnupts, d_plan->spread_params_.pirange, d_plan->grid_dims_[0],
          d_plan->grid_dims_[1], d_plan->grid_dims_[2]);
        break;
    }
  } else {
    TF_CHECK_OK(GpuLaunchKernel(
        TrivialGlobalSortIdxKernel,
        num_blocks, threads_per_block, 0, d_plan->device_.stream(),
        d_plan->num_points_, d_plan->idxnupts));
  }

  return 0;
}

int CUSPREAD2D_NUPTSDRIVEN(Plan<GPUDevice, FLT>* d_plan, int blksize) {
  dim3 threadsPerBlock;
  dim3 blocks;

  int kernel_width=d_plan->spread_params_.nspread;   // psi's support in terms of number of cells
  int pirange=d_plan->spread_params_.pirange;
  int *d_idxnupts=d_plan->idxnupts;
  FLT es_c=d_plan->spread_params_.ES_c;
  FLT es_beta=d_plan->spread_params_.ES_beta;
  FLT sigma=d_plan->spread_params_.upsampling_factor;

  CUCPX* d_c = d_plan->c;
  CUCPX* d_fw = d_plan->fine_grid_data_;

  threadsPerBlock.x = 16;
  threadsPerBlock.y = 1;
  blocks.x = (d_plan->num_points_ + threadsPerBlock.x - 1)/threadsPerBlock.x;
  blocks.y = 1;

  switch (d_plan->rank_) {
    case 2:
      if (d_plan->options_.kernel_evaluation_method == KernelEvaluationMethod::HORNER) {
        for (int t=0; t<blksize; t++) {
          Spread_2d_NUptsdriven_Horner<<<blocks, threadsPerBlock>>>(d_plan->points_[0],
            d_plan->points_[1], d_c + t * d_plan->num_points_,
            d_fw + t * d_plan->grid_count_, d_plan->num_points_, kernel_width,
            d_plan->grid_dims_[0], d_plan->grid_dims_[1], sigma, d_idxnupts, pirange);
        }
      } else {
        for (int t=0; t<blksize; t++) {
          Spread_2d_NUptsdriven<<<blocks, threadsPerBlock>>>(
            d_plan->points_[0], d_plan->points_[1],
            d_c + t * d_plan->grid_count_, d_fw + t * d_plan->grid_count_,
            d_plan->num_points_, kernel_width,
            d_plan->grid_dims_[0], d_plan->grid_dims_[1], es_c, es_beta, d_idxnupts, pirange);
        }
      }
      break;
    case 3:
      if (d_plan->options_.kernel_evaluation_method == KernelEvaluationMethod::HORNER) {
        for (int t=0; t<blksize; t++) {
          Spread_3d_NUptsdriven_Horner<<<blocks, threadsPerBlock>>>(d_plan->points_[0],
            d_plan->points_[1], d_plan->points_[2], d_c+t*d_plan->num_points_,
            d_fw+t*d_plan->grid_count_, d_plan->num_points_, kernel_width,
            d_plan->grid_dims_[0], d_plan->grid_dims_[1], d_plan->grid_dims_[2],
            sigma, d_idxnupts,pirange);
        }
      } else {
        for (int t=0; t<blksize; t++) {
          Spread_3d_NUptsdriven<<<blocks, threadsPerBlock>>>(d_plan->points_[0],
            d_plan->points_[1], d_plan->points_[2],
            d_c+t*d_plan->num_points_, d_fw+t*d_plan->grid_count_, d_plan->num_points_, kernel_width, d_plan->grid_dims_[0],
            d_plan->grid_dims_[1], d_plan->grid_dims_[2], es_c, es_beta, 
            d_idxnupts,pirange);
        }
      }
      break;
  }

  return 0;
}

int CUSPREAD2D_SUBPROB_PROP(Plan<GPUDevice, FLT>* d_plan) {
  int num_blocks = (d_plan->num_points_ + 1024 - 1) / 1024;
  int threads_per_block = 1024;

  int maxsubprobsize=d_plan->options_.gpu_max_subproblem_size;

  int bin_size[3];
  bin_size[0] = d_plan->options_.gpu_bin_size.x;
  bin_size[1] = d_plan->options_.gpu_bin_size.y;
  bin_size[2] = d_plan->options_.gpu_bin_size.z;
  if (bin_size[0] < 0 || bin_size[1] < 0 || bin_size[2] < 0) {
    cout<<"error: invalid binsize (binsizex, binsizey) = (";
    return 1; 
  }

  int num_bins[3] = {1, 1, 1};
  int bin_count = 1;
  for (int i = 0; i < d_plan->rank_; i++) {
    num_bins[i] = (d_plan->grid_dims_[i] + bin_size[i] - 1) / bin_size[i];
    bin_count *= num_bins[i];
  }

  int *d_binsize = d_plan->binsize;
  int *d_binstartpts = d_plan->binstartpts;
  int *d_sortidx = d_plan->sortidx;
  int *d_numsubprob = d_plan->numsubprob;
  int *d_subprobstartpts = d_plan->subprobstartpts;
  int *d_idxnupts = d_plan->idxnupts;

  int *d_subprob_to_bin = NULL;

  int pirange=d_plan->spread_params_.pirange;

  // This may not be necessary.
  d_plan->device_.synchronize();

  // Calculate bin sizes.
  d_plan->device_.memset(d_plan->binsize, 0, bin_count * sizeof(int));
  switch (d_plan->rank_) {
    case 2:
      TF_CHECK_OK(GpuLaunchKernel(
          CalcBinSizeNoGhost2DKernel, num_blocks, threads_per_block, 0,
          d_plan->device_.stream(), d_plan->num_points_, d_plan->grid_dims_[0],
          d_plan->grid_dims_[1], bin_size[0], bin_size[1],
          num_bins[0], num_bins[1], d_binsize, d_plan->points_[0],
          d_plan->points_[1], d_sortidx, pirange));
      break;
    case 3:
      CalcBinSizeNoGhost3DKernel<<<num_blocks, threads_per_block>>>(
        d_plan->num_points_, d_plan->grid_dims_[0], d_plan->grid_dims_[1],
        d_plan->grid_dims_[2], bin_size[0],
        bin_size[1], bin_size[2], num_bins[0], num_bins[1], num_bins[2], d_binsize,
        d_plan->points_[0], d_plan->points_[1], d_plan->points_[2], d_sortidx, pirange);
      break;
  }

  thrust::device_ptr<int> d_ptr(d_binsize);
  thrust::device_ptr<int> d_result(d_binstartpts);
  thrust::exclusive_scan(d_ptr, d_ptr + bin_count, d_result);

  switch (d_plan->rank_) {
    case 2:
      TF_CHECK_OK(GpuLaunchKernel(
          CalcInvertofGlobalSortIdx2DKernel, num_blocks, threads_per_block, 0,
          d_plan->device_.stream(), d_plan->num_points_, bin_size[0], bin_size[1], num_bins[0],
          num_bins[1], d_binstartpts, d_sortidx, d_plan->points_[0],
          d_plan->points_[1], d_idxnupts, pirange, d_plan->grid_dims_[0],
          d_plan->grid_dims_[1]));

      TF_CHECK_OK(GpuLaunchKernel(
          CalcSubProb_2d, num_blocks, threads_per_block, 0,
          d_plan->device_.stream(), d_binsize, d_numsubprob, maxsubprobsize,
          bin_count));
      break;
    case 3:
      CalcInvertofGlobalSortIdx3DKernel<<<num_blocks, threads_per_block>>>(
        d_plan->num_points_, bin_size[0],
        bin_size[1], bin_size[2], num_bins[0], num_bins[1], num_bins[2],
        d_binstartpts, d_sortidx, d_plan->points_[0], d_plan->points_[1],
        d_plan->points_[2], d_idxnupts, pirange, d_plan->grid_dims_[0],
        d_plan->grid_dims_[1], d_plan->grid_dims_[2]);
    
      CalcSubProb_3d_v2<<<num_blocks, threads_per_block>>>(d_binsize,d_numsubprob,
          maxsubprobsize, bin_count);
      break;
  }

  d_ptr    = thrust::device_pointer_cast(d_numsubprob);
  d_result = thrust::device_pointer_cast(d_subprobstartpts+1);
  thrust::inclusive_scan(d_ptr, d_ptr + bin_count, d_result);
  checkCudaErrors(cudaMemset(d_subprobstartpts,0,sizeof(int)));

  int totalnumsubprob;
  checkCudaErrors(cudaMemcpy(&totalnumsubprob,&d_subprobstartpts[bin_count],
    sizeof(int),cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMalloc(&d_subprob_to_bin,totalnumsubprob*sizeof(int)));

  num_blocks = (bin_count + 1024 - 1) / 1024;
  threads_per_block = 1024;

  switch (d_plan->rank_) {
    case 2:
      TF_CHECK_OK(GpuLaunchKernel(
          MapBintoSubProb_2d, num_blocks, threads_per_block, 0,
          d_plan->device_.stream(), d_subprob_to_bin, d_subprobstartpts,
          d_numsubprob, bin_count));
      break;
    case 3:
      MapBintoSubProb_3d_v2<<<(num_bins[0]*num_bins[1]+1024-1)/1024, 1024>>>(
          d_subprob_to_bin, d_subprobstartpts, d_numsubprob, bin_count);
      break;
  }

  assert(d_subprob_to_bin != NULL);
  if (d_plan->subprob_to_bin != NULL) cudaFree(d_plan->subprob_to_bin);
  d_plan->subprob_to_bin = d_subprob_to_bin;
  assert(d_plan->subprob_to_bin != NULL);
  d_plan->totalnumsubprob = totalnumsubprob;

  return 0;
}

int CUSPREAD2D_SUBPROB(Plan<GPUDevice, FLT>* d_plan, int blksize) {
  int kernel_width = d_plan->spread_params_.nspread;// psi's support in terms of number of cells
  FLT es_c=d_plan->spread_params_.ES_c;
  FLT es_beta=d_plan->spread_params_.ES_beta;
  int maxsubprobsize=d_plan->options_.gpu_max_subproblem_size;

  // assume that bin_size_x > kernel_width/2;
  int bin_size[3];
  bin_size[0] = d_plan->options_.gpu_bin_size.x;
  bin_size[1] = d_plan->options_.gpu_bin_size.y;
  bin_size[2] = d_plan->options_.gpu_bin_size.z;

  int num_bins[3] = {1, 1, 1};
  int bin_count = 1;
  for (int i = 0; i < d_plan->rank_; i++) {
    num_bins[i] = (d_plan->grid_dims_[i] + bin_size[i] - 1) / bin_size[i];
    bin_count *= num_bins[i];
  }

  CUCPX* d_c = d_plan->c;
  CUCPX* d_fw = d_plan->fine_grid_data_;

  int *d_binsize = d_plan->binsize;
  int *d_binstartpts = d_plan->binstartpts;
  int *d_numsubprob = d_plan->numsubprob;
  int *d_subprobstartpts = d_plan->subprobstartpts;
  int *d_idxnupts = d_plan->idxnupts;

  int totalnumsubprob=d_plan->totalnumsubprob;
  int *d_subprob_to_bin = d_plan->subprob_to_bin;

  int pirange=d_plan->spread_params_.pirange;

  FLT sigma=d_plan->options_.upsampling_factor;

  // GPU kernel configuration.
  int num_blocks = totalnumsubprob;
  int threads_per_block = 256;
  size_t shared_memory_size = sizeof(CUCPX);
  for (int i = 0; i < d_plan->rank_; i++) {
    shared_memory_size *= (bin_size[i] + 2 * ((kernel_width + 1) / 2));
  }
  if (shared_memory_size > d_plan->device_.sharedMemPerBlock()) {
    cout<<"error: not enough shared memory"<<endl;
    return 1;
  }

  switch (d_plan->rank_) {
    case 2:
      if (d_plan->options_.kernel_evaluation_method == KernelEvaluationMethod::HORNER) {
        for (int t=0; t<blksize; t++) {
          Spread_2d_Subprob_Horner<<<num_blocks, threads_per_block,
            shared_memory_size>>>(d_plan->points_[0], d_plan->points_[1],
              d_c+t*d_plan->num_points_, d_fw+t*d_plan->grid_count_, d_plan->num_points_,
              kernel_width, d_plan->grid_dims_[0], d_plan->grid_dims_[1], sigma, d_binstartpts,
            d_binsize, bin_size[0],
            bin_size[1], d_subprob_to_bin, d_subprobstartpts,
            d_numsubprob, maxsubprobsize,num_bins[0],num_bins[1],
            d_idxnupts, pirange);
        }
      } else {
        for (int t=0; t<blksize; t++) {
          Spread_2d_Subprob<<<num_blocks, threads_per_block, shared_memory_size>>>(
            d_plan->points_[0], d_plan->points_[1], d_c+t*d_plan->num_points_,
            d_fw+t*d_plan->grid_count_, d_plan->num_points_, kernel_width,
            d_plan->grid_dims_[0], d_plan->grid_dims_[1],
            es_c, es_beta, sigma,d_binstartpts, d_binsize, bin_size[0],
            bin_size[1], d_subprob_to_bin, d_subprobstartpts,
            d_numsubprob, maxsubprobsize, num_bins[0], num_bins[1],
            d_idxnupts, pirange);
        }
      }
      break;
    case 3:
      for (int t=0; t<blksize; t++) {
        if (d_plan->options_.kernel_evaluation_method == KernelEvaluationMethod::HORNER) {
          Spread_3d_Subprob_Horner<<<num_blocks, threads_per_block,
            shared_memory_size>>>(d_plan->points_[0], d_plan->points_[1],
              d_plan->points_[2], d_c+t*d_plan->num_points_,
              d_fw+t*d_plan->grid_count_, 
              d_plan->num_points_, kernel_width, d_plan->grid_dims_[0],
              d_plan->grid_dims_[1], d_plan->grid_dims_[2], sigma,
              d_binstartpts, d_binsize, bin_size[0],
              bin_size[1], bin_size[2], d_subprob_to_bin, d_subprobstartpts,
            d_numsubprob, maxsubprobsize,num_bins[0], num_bins[1], num_bins[2],
            d_idxnupts,pirange);
        } else {
          Spread_3d_Subprob<<<num_blocks, threads_per_block,
            shared_memory_size>>>(d_plan->points_[0], d_plan->points_[1],
              d_plan->points_[2], d_c+t*d_plan->num_points_,
              d_fw+t*d_plan->grid_count_, 
              d_plan->num_points_, kernel_width, d_plan->grid_dims_[0],
              d_plan->grid_dims_[1], d_plan->grid_dims_[2], es_c, es_beta,
              d_binstartpts, d_binsize, 
              bin_size[0], bin_size[1], bin_size[2], d_subprob_to_bin, 
            d_subprobstartpts,d_numsubprob, maxsubprobsize,num_bins[0], 
            num_bins[1], num_bins[2],d_idxnupts,pirange);
        }
      }
      break;
  }

  return 0;
}

int CUINTERP2D_NUPTSDRIVEN(int nf1, int nf2, int M, Plan<GPUDevice, FLT>* d_plan, int blksize) {
	dim3 threadsPerBlock;
	dim3 blocks;

	int kernel_width=d_plan->spread_params_.nspread;   // psi's support in terms of number of cells
	FLT es_c=d_plan->spread_params_.ES_c;
	FLT es_beta=d_plan->spread_params_.ES_beta;
	FLT sigma = d_plan->options_.upsampling_factor;
	int pirange=d_plan->spread_params_.pirange;
	int *d_idxnupts=d_plan->idxnupts;

	FLT* d_kx = d_plan->kx;
	FLT* d_ky = d_plan->ky;
	CUCPX* d_c = d_plan->c;
	CUCPX* d_fw = d_plan->fine_grid_data_;

	threadsPerBlock.x = 32;
	threadsPerBlock.y = 1;
	blocks.x = (M + threadsPerBlock.x - 1)/threadsPerBlock.x;
	blocks.y = 1;

	if (d_plan->options_.kernel_evaluation_method == KernelEvaluationMethod::HORNER) {
		for (int t=0; t<blksize; t++) {
			Interp_2d_NUptsdriven_Horner<<<blocks, threadsPerBlock>>>(d_kx, 
				d_ky, d_c+t * d_plan->num_points_, d_fw+t*d_plan->grid_count_,
        d_plan->num_points_, kernel_width, d_plan->grid_dims_[0], d_plan->grid_dims_[1], sigma, 
				d_idxnupts, pirange);
		}
	}else{
		for (int t=0; t<blksize; t++) {
			Interp_2d_NUptsdriven<<<blocks, threadsPerBlock>>>(d_kx, d_ky, 
				d_c+t * d_plan->num_points_, d_fw+t*d_plan->grid_count_,
        d_plan->num_points_, kernel_width, d_plan->grid_dims_[0], d_plan->grid_dims_[1],
        es_c, es_beta,  d_idxnupts, pirange);
		}
	}

	return 0;
}

int CUINTERP2D_SUBPROB(int nf1, int nf2, int M, Plan<GPUDevice, FLT>* d_plan, int blksize) {
	int kernel_width=d_plan->spread_params_.nspread;   // psi's support in terms of number of cells
	FLT es_c=d_plan->spread_params_.ES_c;
	FLT es_beta=d_plan->spread_params_.ES_beta;
	int maxsubprobsize=d_plan->options_.gpu_max_subproblem_size;

	// assume that bin_size_x > kernel_width/2;
  int bin_size[3];
  bin_size[0] = d_plan->options_.gpu_bin_size.x;
  bin_size[1] = d_plan->options_.gpu_bin_size.y;
  bin_size[2] = d_plan->options_.gpu_bin_size.z;

  int num_bins[3] = {1, 1, 1};
  int bin_count = 1;
  for (int i = 0; i < d_plan->rank_; i++) {
    num_bins[i] = (d_plan->grid_dims_[i] + bin_size[i] - 1) / bin_size[i];
    bin_count *= num_bins[i];
  }

	FLT* d_kx = d_plan->kx;
	FLT* d_ky = d_plan->ky;
	CUCPX* d_c = d_plan->c;
	CUCPX* d_fw = d_plan->fine_grid_data_;

	int *d_binsize = d_plan->binsize;
	int *d_binstartpts = d_plan->binstartpts;
	int *d_numsubprob = d_plan->numsubprob;
	int *d_subprobstartpts = d_plan->subprobstartpts;
	int *d_idxnupts = d_plan->idxnupts;
	int *d_subprob_to_bin = d_plan->subprob_to_bin;
	int totalnumsubprob=d_plan->totalnumsubprob;
	int pirange=d_plan->spread_params_.pirange;

	FLT sigma=d_plan->options_.upsampling_factor;

  // GPU kernel configuration.
  int num_blocks = totalnumsubprob;
  int threads_per_block = 256;
  size_t shared_memory_size = sizeof(CUCPX);
  for (int i = 0; i < d_plan->rank_; i++) {
    shared_memory_size *= (bin_size[i] + 2 * ((kernel_width + 1) / 2));
  }
  if (shared_memory_size > d_plan->device_.sharedMemPerBlock()) {
    cout<<"error: not enough shared memory"<<endl;
    return 1;
  }

	if (d_plan->options_.kernel_evaluation_method == KernelEvaluationMethod::HORNER) {
		for (int t=0; t<blksize; t++) {
			Interp_2d_Subprob_Horner<<<num_blocks, threads_per_block, shared_memory_size>>>(
					d_kx, d_ky, d_c+t*d_plan->num_points_,
					d_fw+t*d_plan->grid_count_, d_plan->num_points_, kernel_width,
          d_plan->grid_dims_[0], d_plan->grid_dims_[1], sigma,
					d_binstartpts, d_binsize,
					bin_size[0], bin_size[1],
					d_subprob_to_bin, d_subprobstartpts,
					d_numsubprob, maxsubprobsize,
					num_bins[0], num_bins[1], d_idxnupts, pirange);
		}
	} else {
		for (int t=0; t<blksize; t++) {
			Interp_2d_Subprob<<<num_blocks, threads_per_block, shared_memory_size>>>(
					d_kx, d_ky, d_c+t*d_plan->num_points_,
					d_fw+t*d_plan->grid_count_, d_plan->num_points_, kernel_width,
          d_plan->grid_dims_[0], d_plan->grid_dims_[1],
					es_c, es_beta, sigma,
					d_binstartpts, d_binsize,
					bin_size[0], bin_size[1],
					d_subprob_to_bin, d_subprobstartpts,
					d_numsubprob, maxsubprobsize,
					num_bins[0], num_bins[1], d_idxnupts, pirange);
		}
	}

	return 0;
}

int CUSPREAD2D_PAUL_PROP(Plan<GPUDevice, FLT>* d_plan)
{
  // TODO: unimplemented error.
	return 1;
}

int CUSPREAD2D_PAUL(Plan<GPUDevice, FLT>* d_plan, int blksize)
{
  // TODO: unimplemented error.
	return 1;
}

int CUSPREAD3D_BLOCKGATHER_PROP(
  Plan<GPUDevice, FLT>* d_plan)
{
  // TODO: raise not implemented error
  return 1;
}

int CUSPREAD3D_BLOCKGATHER(
  Plan<GPUDevice, FLT>* d_plan, int blksize)
{
  // TODO: raise not implemented error
  return 1;
}

#include "spread3d_wrapper.cu"

#endif // GOOGLE_CUDA
