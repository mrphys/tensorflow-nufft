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

#include <tensorflow_nufft/cc/kernels/finufft/gpu/contrib/cuda_samples/helper_cuda.h>
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


int CUFINUFFT_INTERP2D(int nf1, int nf2, CUCPX* d_fw, int M, 
	FLT *d_kx, FLT *d_ky, CUCPX *d_c, Plan<GPUDevice, FLT>* d_plan)
/*
	This c function is written for only doing 2D interpolation. See 
	test/interp2d_test.cu for usage.

	Melody Shih 07/25/19
	not allocate,transfer and free memories on gpu. Shih 09/24/20
*/
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	d_plan->nf1 = nf1;
	d_plan->nf2 = nf2;
	d_plan->M = M;
	d_plan->maxbatchsize = 1;

	d_plan->kx = d_kx;
	d_plan->ky = d_ky;
	d_plan->c  = d_c;
	d_plan->fw = d_fw;

	int ier;
	cudaEventRecord(start);
	ier = ALLOCGPUMEM2D_PLAN(d_plan);
	ier = ALLOCGPUMEM2D_NUPTS(d_plan);
	if (d_plan->options.gpu_spread_method == GpuSpreadMethod::NUPTS_DRIVEN) {
		ier = CUSPREAD2D_NUPTSDRIVEN_PROP(nf1,nf2,M,d_plan);
		if (ier != 0 ) {
			printf("error: cuspread2d_subprob_prop, method(%d)\n", 
				d_plan->options.gpu_spread_method);
			return ier;
		}
	}
	if (d_plan->options.gpu_spread_method == GpuSpreadMethod::SUBPROBLEM) {
		ier = CUSPREAD2D_SUBPROB_PROP(nf1,nf2,M,d_plan);
		if (ier != 0 ) {
			printf("error: cuspread2d_subprob_prop, method(%d)\n", 
				d_plan->options.gpu_spread_method);
			return ier;
		}
	}
#ifdef TIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Obtain Interp Prop\t %.3g ms\n", d_plan->options.gpu_spread_method, 
		milliseconds);
#endif
	cudaEventRecord(start);
	ier = CUINTERP2D(d_plan,1);
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Interp (%d)\t\t %.3g ms\n", d_plan->options.gpu_spread_method, 
		milliseconds);
#endif
	cudaEventRecord(start);
	FREEGPUMEMORY2D(d_plan);
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] Free GPU memory\t %.3g ms\n", milliseconds);
#endif
	return ier;
}

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
	switch (d_plan->options.gpu_spread_method)
	{
		case GpuSpreadMethod::NUPTS_DRIVEN:
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
		case GpuSpreadMethod::SUBPROBLEM:
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

	int ns=d_plan->spopts.nspread;   // psi's support in terms of number of cells
	FLT es_c=d_plan->spopts.ES_c;
	FLT es_beta=d_plan->spopts.ES_beta;
	FLT sigma = d_plan->options.upsampling_factor;
	int pirange=d_plan->spopts.pirange;
	int *d_idxnupts=d_plan->idxnupts;

	FLT* d_kx = d_plan->kx;
	FLT* d_ky = d_plan->ky;
	CUCPX* d_c = d_plan->c;
	CUCPX* d_fw = d_plan->fw;

	threadsPerBlock.x = 32;
	threadsPerBlock.y = 1;
	blocks.x = (M + threadsPerBlock.x - 1)/threadsPerBlock.x;
	blocks.y = 1;

	cudaEventRecord(start);
	if (d_plan->options.kernel_evaluation_method == KernelEvaluationMethod::HORNER) {
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
		milliseconds, d_plan->options.kernel_evaluation_method);
#endif
	return 0;
}

int CUINTERP2D_SUBPROB(int nf1, int nf2, int M, Plan<GPUDevice, FLT>* d_plan,
	int blksize)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int ns=d_plan->spopts.nspread;   // psi's support in terms of number of cells
	FLT es_c=d_plan->spopts.ES_c;
	FLT es_beta=d_plan->spopts.ES_beta;
	int maxsubprobsize=d_plan->options.gpu_max_subproblem_size;

	// assume that bin_size_x > ns/2;
	int bin_size_x=d_plan->options.gpu_bin_size.x;
	int bin_size_y=d_plan->options.gpu_bin_size.y;
	int numbins[2];
	numbins[0] = ceil((FLT) nf1/bin_size_x);
	numbins[1] = ceil((FLT) nf2/bin_size_y);
#ifdef INFO
	cout<<"[info  ] Dividing the uniform grids to bin size["
		<<d_plan->options.gpu_bin_size.x<<"x"<<d_plan->options.gpu_bin_size.y<<"]"<<endl;
	cout<<"[info  ] numbins = ["<<numbins[0]<<"x"<<numbins[1]<<"]"<<endl;
#endif

	FLT* d_kx = d_plan->kx;
	FLT* d_ky = d_plan->ky;
	CUCPX* d_c = d_plan->c;
	CUCPX* d_fw = d_plan->fw;

	int *d_binsize = d_plan->binsize;
	int *d_binstartpts = d_plan->binstartpts;
	int *d_numsubprob = d_plan->numsubprob;
	int *d_subprobstartpts = d_plan->subprobstartpts;
	int *d_idxnupts = d_plan->idxnupts;
	int *d_subprob_to_bin = d_plan->subprob_to_bin;
	int totalnumsubprob=d_plan->totalnumsubprob;
	int pirange=d_plan->spopts.pirange;

	FLT sigma=d_plan->options.upsampling_factor;
	cudaEventRecord(start);
	size_t sharedplanorysize = (bin_size_x+2*ceil(ns/2.0))*(bin_size_y+2*
		ceil(ns/2.0))*sizeof(CUCPX);
	if (sharedplanorysize > 49152) {
		cout<<"error: not enough shared memory"<<endl;
		return 1;
	}

	if (d_plan->options.kernel_evaluation_method == KernelEvaluationMethod::HORNER) {
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
		milliseconds, d_plan->options.kernel_evaluation_method);
#endif
	return 0;
}
