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
	int* sortidx, int pirange)
{
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

__global__ void CalcInvertofGlobalSortIdx2DKernel(int M, int bin_size_x, int bin_size_y, 
	int nbinx,int nbiny, int* bin_startpts, int* sortidx, FLT *x, FLT *y, 
	int* index, int pirange, int nf1, int nf2)
{
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

#ifdef SINGLE
#define TrivialGlobalSortIdx2DKernel TrivialGlobalSortIdx2DKernel_S
#else
#define TrivialGlobalSortIdx2DKernel TrivialGlobalSortIdx2DKernel_D
#endif
__global__ void TrivialGlobalSortIdx2DKernel(int M, int* index)
{
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
  int nf1 = d_plan->nf1;
  int nf2 = d_plan->nf2;
  int M = d_plan->M;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int ier;
  switch(d_plan->options_.spread_method)
  {
    case SpreadMethod::NUPTS_DRIVEN:
      {
        cudaEventRecord(start);
        ier = CUSPREAD2D_NUPTSDRIVEN(nf1, nf2, M, d_plan, blksize);
        if (ier != 0 ) {
          cout<<"error: cnufftspread2d_gpu_nuptsdriven"<<endl;
          return 1;
        }
      }
      break;
    case SpreadMethod::SUBPROBLEM:
      {
        cudaEventRecord(start);
        ier = CUSPREAD2D_SUBPROB(nf1, nf2, M, d_plan, blksize);
        if (ier != 0 ) {
          cout<<"error: cnufftspread2d_gpu_subprob"<<endl;
          return 1;
        }
      }
      break;
    case SpreadMethod::PAUL:
      {
        cudaEventRecord(start);
        ier = CUSPREAD2D_PAUL(nf1, nf2, M, d_plan, blksize);
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

int CUSPREAD2D_NUPTSDRIVEN_PROP(int nf1, int nf2, int M, Plan<GPUDevice, FLT>* d_plan)
{
  int num_blocks = (M + 1024 - 1) / 1024;
  int threads_per_block = 1024;

  if (d_plan->spread_params_.sort_points == SortPoints::YES) {

    int bin_size[2];
    bin_size[0] = d_plan->options_.gpu_bin_size.x;
    bin_size[1] = d_plan->options_.gpu_bin_size.y;
    if (bin_size[0] < 0 || bin_size[1] < 0) {
      cout << "error: invalid binsize (binsizex, binsizey) = (";
      cout << bin_size[0] << "," << bin_size[1] << ")" << endl;
      return 1; 
    }

    int num_bins[2];
    num_bins[0] = ceil((FLT) nf1 / bin_size[0]);
    num_bins[1] = ceil((FLT) nf2 / bin_size[1]);

    FLT*   d_kx = d_plan->kx;
    FLT*   d_ky = d_plan->ky;

    int *d_binsize = d_plan->binsize;
    int *d_binstartpts = d_plan->binstartpts;
    int *d_sortidx = d_plan->sortidx;
    int *d_idxnupts = d_plan->idxnupts;

    int pirange = d_plan->spread_params_.pirange;

    // Synchronize device before we start. This is essential! Otherwise the
    // next kernel could read the wrong (kx, ky, kz) values.
    d_plan->device_.synchronize();

    d_plan->device_.memset(d_binsize, 0, num_bins[0] * num_bins[1] * sizeof(int));
    
    TF_CHECK_OK(GpuLaunchKernel(
        CalcBinSizeNoGhost2DKernel,
        num_blocks, threads_per_block, 0, d_plan->device_.stream(),
        M, nf1, nf2, bin_size[0], bin_size[1], num_bins[0], num_bins[1],
        d_binsize, d_kx, d_ky, d_sortidx, pirange));

    int n=num_bins[0]*num_bins[1];
    thrust::device_ptr<int> d_ptr(d_binsize);
    thrust::device_ptr<int> d_result(d_binstartpts);
    thrust::exclusive_scan(d_ptr, d_ptr + n, d_result);

    TF_CHECK_OK(GpuLaunchKernel(
        CalcInvertofGlobalSortIdx2DKernel,
        num_blocks, threads_per_block, 0, d_plan->device_.stream(),
        M, bin_size[0], bin_size[1], num_bins[0], num_bins[1],
        d_binstartpts, d_sortidx, d_kx, d_ky,
        d_idxnupts, pirange, nf1, nf2));
  } else {
    TF_CHECK_OK(GpuLaunchKernel(
        TrivialGlobalSortIdx2DKernel,
        num_blocks, threads_per_block, 0, d_plan->device_.stream(),
        M, d_plan->idxnupts));
  }
  return 0;
}

int CUSPREAD2D_NUPTSDRIVEN(int nf1, int nf2, int M, Plan<GPUDevice, FLT>* d_plan,
  int blksize)
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  dim3 threadsPerBlock;
  dim3 blocks;

  int ns=d_plan->spread_params_.nspread;   // psi's support in terms of number of cells
  int pirange=d_plan->spread_params_.pirange;
  int *d_idxnupts=d_plan->idxnupts;
  FLT es_c=d_plan->spread_params_.ES_c;
  FLT es_beta=d_plan->spread_params_.ES_beta;
  FLT sigma=d_plan->spread_params_.upsampling_factor;

  FLT* d_kx = d_plan->kx;
  FLT* d_ky = d_plan->ky;
  CUCPX* d_c = d_plan->c;
  CUCPX* d_fw = d_plan->fine_grid_data_;

  threadsPerBlock.x = 16;
  threadsPerBlock.y = 1;
  blocks.x = (M + threadsPerBlock.x - 1)/threadsPerBlock.x;
  blocks.y = 1;

  if (d_plan->options_.kernel_evaluation_method == KernelEvaluationMethod::HORNER) {
    for (int t=0; t<blksize; t++) {
      Spread_2d_NUptsdriven_Horner<<<blocks, threadsPerBlock>>>(d_kx,
        d_ky, d_c+t*M, d_fw+t*nf1*nf2, M, ns, nf1, nf2, sigma,
        d_idxnupts, pirange);
    }
  } else {
    for (int t=0; t<blksize; t++) {
      Spread_2d_NUptsdriven<<<blocks, threadsPerBlock>>>(d_kx, d_ky,
        d_c+t*M, d_fw+t*nf1*nf2, M, ns, nf1, nf2, es_c, es_beta,
        d_idxnupts, pirange);
    }
  }

  return 0;
}
int CUSPREAD2D_SUBPROB_PROP(int nf1, int nf2, int M, Plan<GPUDevice, FLT>* d_plan)
/*
  This function determines the properties for spreading that are independent
  of the strength of the nodes,  only relates to the locations of the nodes,
  which only needs to be done once.
*/
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int maxsubprobsize=d_plan->options_.gpu_max_subproblem_size;
  int bin_size_x=d_plan->options_.gpu_bin_size.x;
  int bin_size_y=d_plan->options_.gpu_bin_size.y;
  if (bin_size_x < 0 || bin_size_y < 0) {
    cout<<"error: invalid binsize (binsizex, binsizey) = (";
    cout<<bin_size_x<<","<<bin_size_y<<")"<<endl;
    return 1; 
  }
  int num_bins[2];
  num_bins[0] = ceil((FLT) nf1/bin_size_x);
  num_bins[1] = ceil((FLT) nf2/bin_size_y);

  FLT*   d_kx = d_plan->kx;
  FLT*   d_ky = d_plan->ky;

  int *d_binsize = d_plan->binsize;
  int *d_binstartpts = d_plan->binstartpts;
  int *d_sortidx = d_plan->sortidx;
  int *d_numsubprob = d_plan->numsubprob;
  int *d_subprobstartpts = d_plan->subprobstartpts;
  int *d_idxnupts = d_plan->idxnupts;

  int *d_subprob_to_bin = NULL;

  int pirange=d_plan->spread_params_.pirange;

  // Synchronize device before we start. This is essential! Otherwise the
  // next kernel could read the wrong (kx, ky, kz) values.
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaMemset(d_binsize,0,num_bins[0]*num_bins[1]*sizeof(int)));
  CalcBinSizeNoGhost2DKernel<<<(M+1024-1)/1024, 1024>>>(M,nf1,nf2,bin_size_x,
    bin_size_y,num_bins[0],num_bins[1],d_binsize,d_kx,d_ky,d_sortidx,pirange);

  int n=num_bins[0]*num_bins[1];
  thrust::device_ptr<int> d_ptr(d_binsize);
  thrust::device_ptr<int> d_result(d_binstartpts);
  thrust::exclusive_scan(d_ptr, d_ptr + n, d_result);

  CalcInvertofGlobalSortIdx2DKernel<<<(M+1024-1)/1024,1024>>>(M,bin_size_x,
    bin_size_y,num_bins[0],num_bins[1],d_binstartpts,d_sortidx,d_kx,d_ky,
    d_idxnupts,pirange,nf1,nf2);

  CalcSubProb_2d<<<(M+1024-1)/1024, 1024>>>(d_binsize,d_numsubprob,
    maxsubprobsize,num_bins[0]*num_bins[1]);

  d_ptr    = thrust::device_pointer_cast(d_numsubprob);
  d_result = thrust::device_pointer_cast(d_subprobstartpts+1);
  thrust::inclusive_scan(d_ptr, d_ptr + n, d_result);
  checkCudaErrors(cudaMemset(d_subprobstartpts,0,sizeof(int)));

  int totalnumsubprob;
  checkCudaErrors(cudaMemcpy(&totalnumsubprob,&d_subprobstartpts[n],
    sizeof(int),cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMalloc(&d_subprob_to_bin,totalnumsubprob*sizeof(int)));
  MapBintoSubProb_2d<<<(num_bins[0]*num_bins[1]+1024-1)/1024, 1024>>>(
      d_subprob_to_bin,d_subprobstartpts,d_numsubprob,num_bins[0]*num_bins[1]);
  assert(d_subprob_to_bin != NULL);
        if (d_plan->subprob_to_bin != NULL) cudaFree(d_plan->subprob_to_bin);
  d_plan->subprob_to_bin = d_subprob_to_bin;
  assert(d_plan->subprob_to_bin != NULL);
  d_plan->totalnumsubprob = totalnumsubprob;

  return 0;
}

int CUSPREAD2D_SUBPROB(int nf1, int nf2, int M, Plan<GPUDevice, FLT>* d_plan,
  int blksize)
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int ns=d_plan->spread_params_.nspread;// psi's support in terms of number of cells
  FLT es_c=d_plan->spread_params_.ES_c;
  FLT es_beta=d_plan->spread_params_.ES_beta;
  int maxsubprobsize=d_plan->options_.gpu_max_subproblem_size;

  // assume that bin_size_x > ns/2;
  int bin_size_x=d_plan->options_.gpu_bin_size.x;
  int bin_size_y=d_plan->options_.gpu_bin_size.y;
  int num_bins[2];
  num_bins[0] = ceil((FLT) nf1/bin_size_x);
  num_bins[1] = ceil((FLT) nf2/bin_size_y);
#ifdef INFO
  cout<<"[info  ] Dividing the uniform grids to bin size["
    <<d_plan->options_.gpu_bin_size.x<<"x"<<d_plan->options_.gpu_bin_size.y<<"]"<<endl;
  cout<<"[info  ] num_bins = ["<<num_bins[0]<<"x"<<num_bins[1]<<"]"<<endl;
#endif

  FLT* d_kx = d_plan->kx;
  FLT* d_ky = d_plan->ky;
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
  cudaEventRecord(start);

  size_t sharedplanorysize = (bin_size_x+2*(int)ceil(ns/2.0))*
                 (bin_size_y+2*(int)ceil(ns/2.0))*
                 sizeof(CUCPX);
  if (sharedplanorysize > 49152) {
    cout<<"error: not enough shared memory"<<endl;
    return 1;
  }

  if (d_plan->options_.kernel_evaluation_method == KernelEvaluationMethod::HORNER) {
    for (int t=0; t<blksize; t++) {
      Spread_2d_Subprob_Horner<<<totalnumsubprob, 256,
        sharedplanorysize>>>(d_kx, d_ky, d_c+t*M, d_fw+t*nf1*nf2, M,
        ns, nf1, nf2, sigma, d_binstartpts, d_binsize, bin_size_x,
        bin_size_y, d_subprob_to_bin, d_subprobstartpts,
        d_numsubprob, maxsubprobsize,num_bins[0],num_bins[1],
        d_idxnupts, pirange);
    }
  }else{
    for (int t=0; t<blksize; t++) {
      Spread_2d_Subprob<<<totalnumsubprob, 256, sharedplanorysize>>>(
        d_kx, d_ky, d_c+t*M, d_fw+t*nf1*nf2, M, ns, nf1, nf2,
        es_c, es_beta, sigma,d_binstartpts, d_binsize, bin_size_x,
        bin_size_y, d_subprob_to_bin, d_subprobstartpts,
        d_numsubprob, maxsubprobsize, num_bins[0], num_bins[1],
        d_idxnupts, pirange);
    }
  }
#ifdef SPREADTIME
  float milliseconds = 0;
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("[time  ] \tKernel Spread_2d_Subprob (%d)\t\t%.3g ms\n",
    milliseconds, d_plan->options_.kernel_evaluation_method);
#endif
  return 0;
}

#endif // GOOGLE_CUDA
