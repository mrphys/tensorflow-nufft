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

#include <tensorflow_nufft/third_party/cuda_samples/helper_cuda.h>
#include <iostream>
#include <iomanip>

#include <cuComplex.h>
#include "tensorflow_nufft/cc/kernels/finufft/gpu/cuspreadinterp.h"
#include "tensorflow_nufft/cc/kernels/finufft/gpu/memtransfer.h"
#include <profile.h>
#include "tensorflow_nufft/cc/kernels/nufft_options.h"
#include "tensorflow_nufft/cc/kernels/nufft_plan.h"

using namespace std;
using namespace tensorflow;
using namespace tensorflow::nufft;


int CUINTERP2D(Plan<GPUDevice, FLT>* d_plan, int blksize)
/*
	A wrapper for different interpolation methods. 

	Methods available:
	(1) Non-uniform points driven
	(2) Subproblem

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
	switch (d_plan->options_.spread_method)
	{
		case SpreadMethod::NUPTS_DRIVEN:
			{
				cudaEventRecord(start);
				{
					PROFILE_CUDA_GROUP("Spreading", 6);
					ier = CUINTERP2D_NUPTSDRIVEN(nf1, nf2, M, d_plan, blksize);
					if (ier != 0 ) {
						cout<<"error: cnufftspread2d_gpu_nuptsdriven"<<endl;
						return 1;
					}
				}
			}
			break;
		case SpreadMethod::SUBPROBLEM:
			{
				cudaEventRecord(start);
				ier = CUINTERP2D_SUBPROB(nf1, nf2, M, d_plan, blksize);
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
#ifdef SPREADTIME
	float milliseconds;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout<<"[time  ]"<< " Interp " << milliseconds <<" ms"<<endl;
#endif
	return ier;
}

int CUINTERP2D_NUPTSDRIVEN(int nf1, int nf2, int M, Plan<GPUDevice, FLT>* d_plan,
	int blksize)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 threadsPerBlock;
	dim3 blocks;

	int ns=d_plan->spread_params_.nspread;   // psi's support in terms of number of cells
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

	cudaEventRecord(start);
	if (d_plan->options_.kernel_evaluation_method == KernelEvaluationMethod::HORNER) {
		for (int t=0; t<blksize; t++) {
			Interp_2d_NUptsdriven_Horner<<<blocks, threadsPerBlock>>>(d_kx, 
				d_ky, d_c+t*M, d_fw+t*nf1*nf2, M, ns, nf1, nf2, sigma, 
				d_idxnupts, pirange);
		}
	}else{
		for (int t=0; t<blksize; t++) {
			Interp_2d_NUptsdriven<<<blocks, threadsPerBlock>>>(d_kx, d_ky, 
				d_c+t*M, d_fw+t*nf1*nf2, M, ns, nf1, nf2, es_c, es_beta, 
				d_idxnupts, pirange);
		}
	}
#ifdef SPREADTIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel Interp_2d_NUptsdriven (%d)\t%.3g ms\n", 
		milliseconds, d_plan->options_.kernel_evaluation_method);
#endif
	return 0;
}

int CUINTERP2D_SUBPROB(int nf1, int nf2, int M, Plan<GPUDevice, FLT>* d_plan,
	int blksize)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int ns=d_plan->spread_params_.nspread;   // psi's support in terms of number of cells
	FLT es_c=d_plan->spread_params_.ES_c;
	FLT es_beta=d_plan->spread_params_.ES_beta;
	int maxsubprobsize=d_plan->options_.gpu_max_subproblem_size;

	// assume that bin_size_x > ns/2;
	int bin_size_x=d_plan->options_.gpu_bin_size.x;
	int bin_size_y=d_plan->options_.gpu_bin_size.y;
	int numbins[2];
	numbins[0] = ceil((FLT) nf1/bin_size_x);
	numbins[1] = ceil((FLT) nf2/bin_size_y);
#ifdef INFO
	cout<<"[info  ] Dividing the uniform grids to bin size["
		<<d_plan->options_.gpu_bin_size.x<<"x"<<d_plan->options_.gpu_bin_size.y<<"]"<<endl;
	cout<<"[info  ] numbins = ["<<numbins[0]<<"x"<<numbins[1]<<"]"<<endl;
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
	int *d_subprob_to_bin = d_plan->subprob_to_bin;
	int totalnumsubprob=d_plan->totalnumsubprob;
	int pirange=d_plan->spread_params_.pirange;

	FLT sigma=d_plan->options_.upsampling_factor;
	cudaEventRecord(start);
	size_t sharedplanorysize = (bin_size_x+2*ceil(ns/2.0))*(bin_size_y+2*
		ceil(ns/2.0))*sizeof(CUCPX);
	if (sharedplanorysize > 49152) {
		cout<<"error: not enough shared memory"<<endl;
		return 1;
	}

	if (d_plan->options_.kernel_evaluation_method == KernelEvaluationMethod::HORNER) {
		for (int t=0; t<blksize; t++) {
			Interp_2d_Subprob_Horner<<<totalnumsubprob, 256, sharedplanorysize>>>(
					d_kx, d_ky, d_c+t*M,
					d_fw+t*nf1*nf2, M, ns, nf1, nf2, sigma,
					d_binstartpts, d_binsize,
					bin_size_x, bin_size_y,
					d_subprob_to_bin, d_subprobstartpts,
					d_numsubprob, maxsubprobsize,
					numbins[0], numbins[1], d_idxnupts, pirange);
		}
	} else {
		for (int t=0; t<blksize; t++) {
			Interp_2d_Subprob<<<totalnumsubprob, 256, sharedplanorysize>>>(
					d_kx, d_ky, d_c+t*M,
					d_fw+t*nf1*nf2, M, ns, nf1, nf2,
					es_c, es_beta, sigma,
					d_binstartpts, d_binsize,
					bin_size_x, bin_size_y,
					d_subprob_to_bin, d_subprobstartpts,
					d_numsubprob, maxsubprobsize,
					numbins[0], numbins[1], d_idxnupts, pirange);
		}
	}
#ifdef SPREADTIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel Interp_2d_Subprob (%d)\t\t%.3g ms\n", 
		milliseconds, d_plan->options_.kernel_evaluation_method);
#endif
	return 0;
}