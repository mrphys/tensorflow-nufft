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

using namespace std;

namespace tensorflow {
namespace nufft {

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

int CUINTERP2D_NUPTSDRIVEN(Plan<GPUDevice, FLT>* d_plan, int blksize) {
	dim3 threadsPerBlock;
	dim3 blocks;

	int kernel_width=d_plan->spread_params_.nspread;   // psi's support in terms of number of cells
	FLT es_c=d_plan->spread_params_.ES_c;
	FLT es_beta=d_plan->spread_params_.ES_beta;
	FLT sigma = d_plan->options_.upsampling_factor;
	int pirange=d_plan->spread_params_.pirange;
	int *d_idxnupts=d_plan->idxnupts;

	CUCPX* d_c = d_plan->c;
	CUCPX* d_fw = d_plan->fine_grid_data_;

  switch (d_plan->rank_) {
    case 2:
      threadsPerBlock.x = 32;
      threadsPerBlock.y = 1;
      blocks.x = (d_plan->num_points_ + threadsPerBlock.x - 1)/threadsPerBlock.x;
      blocks.y = 1;

      if (d_plan->options_.kernel_evaluation_method == KernelEvaluationMethod::HORNER) {
        for (int t=0; t<blksize; t++) {
          Interp_2d_NUptsdriven_Horner<<<blocks, threadsPerBlock>>>(
            d_plan->points_[0], d_plan->points_[1], d_c+t * d_plan->num_points_,
            d_fw+t*d_plan->grid_count_, d_plan->num_points_, kernel_width,
            d_plan->grid_dims_[0], d_plan->grid_dims_[1], sigma,  d_idxnupts,
            pirange);
        }
      } else {
        for (int t=0; t<blksize; t++) {
          Interp_2d_NUptsdriven<<<blocks, threadsPerBlock>>>(
            d_plan->points_[0], d_plan->points_[1], 
            d_c+t * d_plan->num_points_, d_fw+t*d_plan->grid_count_,
            d_plan->num_points_, kernel_width, d_plan->grid_dims_[0], d_plan->grid_dims_[1],
            es_c, es_beta,  d_idxnupts, pirange);
        }
      }
      break;
    case 3:
      threadsPerBlock.x = 16;
      threadsPerBlock.y = 1;
      blocks.x = (d_plan->num_points_ + threadsPerBlock.x - 1)/threadsPerBlock.x;
      blocks.y = 1;

      if (d_plan->options_.kernel_evaluation_method == KernelEvaluationMethod::HORNER) {
        for (int t=0; t<blksize; t++) {
          Interp_3d_NUptsdriven_Horner<<<blocks, threadsPerBlock, 0, 0>>>(
              d_plan->points_[0], d_plan->points_[1], d_plan->points_[2],
              d_c + t * d_plan->num_points_, d_fw+t*d_plan->grid_count_,
              d_plan->num_points_, kernel_width, d_plan->grid_dims_[0],
              d_plan->grid_dims_[1], d_plan->grid_dims_[2], sigma, d_idxnupts,
              pirange);
        }
      } else {
        for (int t=0; t<blksize; t++) {
          Interp_3d_NUptsdriven<<<blocks, threadsPerBlock, 0, 0>>>(
              d_plan->points_[0], d_plan->points_[1], d_plan->points_[2],
              d_c + t * d_plan->num_points_, d_fw + t * d_plan->grid_count_,
              d_plan->num_points_, kernel_width, 
              d_plan->grid_dims_[0], d_plan->grid_dims_[1], d_plan->grid_dims_[2],
              es_c, es_beta, d_idxnupts,pirange);
        }
      }
      break;
  }

	return 0;
}

int CUINTERP2D_SUBPROB(Plan<GPUDevice, FLT>* d_plan, int blksize) {
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

  switch (d_plan->rank_) {
    case 2:
      if (d_plan->options_.kernel_evaluation_method == KernelEvaluationMethod::HORNER) {
        for (int t=0; t<blksize; t++) {
          Interp_2d_Subprob_Horner<<<num_blocks, threads_per_block, shared_memory_size>>>(
              d_plan->points_[0], d_plan->points_[1], d_c+t*d_plan->num_points_,
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
              d_plan->points_[0], d_plan->points_[1], d_c + t * d_plan->num_points_,
              d_fw + t * d_plan->grid_count_, d_plan->num_points_, kernel_width,
              d_plan->grid_dims_[0], d_plan->grid_dims_[1],
              es_c, es_beta, sigma,
              d_binstartpts, d_binsize,
              bin_size[0], bin_size[1],
              d_subprob_to_bin, d_subprobstartpts,
              d_numsubprob, maxsubprobsize,
              num_bins[0], num_bins[1], d_idxnupts, pirange);
        }
      }
      break;
    case 3:
      for (int t=0; t<blksize; t++) {
        if (d_plan->options_.kernel_evaluation_method == KernelEvaluationMethod::HORNER) {
          Interp_3d_Subprob_Horner<<<num_blocks, threads_per_block,
            shared_memory_size>>>(
              d_plan->points_[0], d_plan->points_[1], d_plan->points_[2],
              d_c + t * d_plan->num_points_, d_fw + t * d_plan->grid_count_, 
            d_plan->num_points_, kernel_width, d_plan->grid_dims_[0],
            d_plan->grid_dims_[1], d_plan->grid_dims_[2], sigma,
            d_binstartpts, d_binsize, bin_size[0],
            bin_size[1], bin_size[2], d_subprob_to_bin, d_subprobstartpts,
            d_numsubprob, maxsubprobsize,num_bins[0], num_bins[1], num_bins[2],
            d_idxnupts, pirange);
        }else{
          Interp_3d_Subprob<<<num_blocks, threads_per_block,
            shared_memory_size>>>(
              d_plan->points_[0], d_plan->points_[1], d_plan->points_[2],
              d_c + t * d_plan->num_points_, d_fw + t * d_plan->grid_count_, 
            d_plan->num_points_, kernel_width, d_plan->grid_dims_[0], d_plan->grid_dims_[1],
            d_plan->grid_dims_[2], es_c, es_beta, d_binstartpts, d_binsize, 
            bin_size[0], bin_size[1], bin_size[2], d_subprob_to_bin, 
            d_subprobstartpts, d_numsubprob, maxsubprobsize,num_bins[0], 
            num_bins[1], num_bins[2], d_idxnupts, pirange);
        }
      }
      break;
  }

	return 0;
}

int CUSPREAD2D_PAUL(Plan<GPUDevice, FLT>* d_plan, int blksize)
{
  // TODO: unimplemented error.
	return 1;
}

int CUSPREAD3D_BLOCKGATHER(
  Plan<GPUDevice, FLT>* d_plan, int blksize)
{
  // TODO: raise not implemented error
  return 1;
}

int CUSPREAD2D(Plan<GPUDevice, FLT>* d_plan, int blksize) {
  int ier;
  switch(d_plan->options_.spread_method)
  {
    case SpreadMethod::NUPTS_DRIVEN:
      {
        ier = CUSPREAD2D_NUPTSDRIVEN(d_plan, blksize);
        if (ier != 0 ) {
          cout<<"error: cnufftspread2d_gpu_nuptsdriven"<<endl;
          return 1;
        }
      }
      break;
    case SpreadMethod::SUBPROBLEM:
      {
        ier = CUSPREAD2D_SUBPROB(d_plan, blksize);
        if (ier != 0 ) {
          cout<<"error: cnufftspread2d_gpu_subprob"<<endl;
          return 1;
        }
      }
      break;
    case SpreadMethod::PAUL:
      {
        ier = CUSPREAD2D_PAUL(d_plan, blksize);
        if (ier != 0 ) {
          cout<<"error: cnufftspread2d_gpu_paul"<<endl;
          return 1;
        }
      }
      break;
    case SpreadMethod::BLOCK_GATHER:
      {
        ier = CUSPREAD3D_BLOCKGATHER(d_plan, blksize);
        if (ier != 0) {
          cout<<"error: cnufftspread2d_gpu_blockgather"<<endl;
          return 1;
        }
      }
    default:
      cout<<"error: incorrect method, should be 1,2,3"<<endl;
      return 2;
  }

  return ier;
}

int CUINTERP2D(Plan<GPUDevice, FLT>* d_plan, int blksize) {

	int ier;
	switch (d_plan->options_.spread_method)
	{
		case SpreadMethod::NUPTS_DRIVEN:
			{
        ier = CUINTERP2D_NUPTSDRIVEN(d_plan, blksize);
        if (ier != 0 ) {
          cout<<"error: cnufftspread2d_gpu_nuptsdriven"<<endl;
          return 1;
        }
			}
			break;
		case SpreadMethod::SUBPROBLEM:
			{
				ier = CUINTERP2D_SUBPROB(d_plan, blksize);
				if (ier != 0 ) {
					cout<<"error: cuinterp2d_subprob"<<endl;
					return 1;
				}
			}
			break;
		default:
			cout<<"error: incorrect method, should be 1 or 2"<<endl;
			return 2;
	}

	return ier;
}

} // namespace nufft
} // namespace tensorflow

#endif // GOOGLE_CUDA
