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

using namespace std;
using namespace tensorflow;
using namespace tensorflow::nufft;


int CUINTERP3D(Plan<GPUDevice, FLT>* d_plan, int blksize)
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
	int nf3 = d_plan->nf3;
	int M = d_plan->M;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int ier;
	switch(d_plan->options_.gpu_spread_method)
	{
		case GpuSpreadMethod::NUPTS_DRIVEN:
			{
				cudaEventRecord(start);
				{
					PROFILE_CUDA_GROUP("Interpolation", 6);
					ier = CUINTERP3D_NUPTSDRIVEN(nf1, nf2, nf3, M, d_plan, blksize);
					if (ier != 0 ) {
						cout<<"error: cnufftspread3d_gpu_nuptsdriven"<<endl;
						return 1;
					}
				}
			}
			break;
		case GpuSpreadMethod::SUBPROBLEM:
			{
				cudaEventRecord(start);
				{
					PROFILE_CUDA_GROUP("Interpolation", 6);
					ier = CUINTERP3D_SUBPROB(nf1, nf2, nf3, M, d_plan, blksize);
					if (ier != 0 ) {
						cout<<"error: cnufftspread3d_gpu_subprob"<<endl;
						return 1;
					}
				}
			}
			break;
		default:
			cout<<"error: incorrect method, should be 1,2"<<endl;
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


int CUINTERP3D_NUPTSDRIVEN(int nf1, int nf2, int nf3, int M, Plan<GPUDevice, FLT>* d_plan,
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
	FLT sigma=d_plan->spopts.upsampling_factor;
	int pirange=d_plan->spopts.pirange;

	int *d_idxnupts = d_plan->idxnupts;

	FLT* d_kx = d_plan->kx;
	FLT* d_ky = d_plan->ky;
	FLT* d_kz = d_plan->kz;
	CUCPX* d_c = d_plan->c;
	CUCPX* d_fw = d_plan->fine_grid_data_;

	threadsPerBlock.x = 16;
	threadsPerBlock.y = 1;
	blocks.x = (M + threadsPerBlock.x - 1)/threadsPerBlock.x;
	blocks.y = 1;

	cudaEventRecord(start);
	if (d_plan->options_.kernel_evaluation_method == KernelEvaluationMethod::HORNER) {
		for (int t=0; t<blksize; t++) {
			Interp_3d_NUptsdriven_Horner<<<blocks, threadsPerBlock, 0, 
				0>>>(d_kx, d_ky, d_kz, d_c+t*M, 
				d_fw+t*nf1*nf2*nf3, M, ns, nf1, nf2, nf3, sigma, d_idxnupts,
				pirange);
		}
	}else{
		for (int t=0; t<blksize; t++) {
			Interp_3d_NUptsdriven<<<blocks, threadsPerBlock, 0, 0 
				>>>(d_kx, d_ky, d_kz, d_c+t*M, d_fw+t*nf1*nf2*nf3, M, ns, 
				nf1, nf2, nf3,es_c, es_beta, d_idxnupts,pirange);
		}
	}
#ifdef SPREADTIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel Interp_3d_NUptsdriven (%d) \t%.3g ms\n", 
		milliseconds, d_plan->options_.kernel_evaluation_method);
#endif
	return 0;
}

int CUINTERP3D_SUBPROB(int nf1, int nf2, int nf3, int M, Plan<GPUDevice, FLT>* d_plan,
	int blksize)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int ns=d_plan->spopts.nspread;   // psi's support in terms of number of cells
	int maxsubprobsize=d_plan->options_.gpu_max_subproblem_size;

	// assume that bin_size_x > ns/2;
	int bin_size_x=d_plan->options_.gpu_bin_size.x;
	int bin_size_y=d_plan->options_.gpu_bin_size.y;
	int bin_size_z=d_plan->options_.gpu_bin_size.z;
	int numbins[3];
	numbins[0] = ceil((FLT) nf1/bin_size_x);
	numbins[1] = ceil((FLT) nf2/bin_size_y);
	numbins[2] = ceil((FLT) nf3/bin_size_z);
#ifdef INFO
	cout<<"[info  ] Dividing the uniform grids to bin size["
		<<d_plan->options_.gpu_bin_size.x<<"x"<<d_plan->options_.gpu_bin_size.y<<"x"<<d_plan->options_.gpu_bin_size.z<<"]"<<endl;
	cout<<"[info  ] numbins = ["<<numbins[0]<<"x"<<numbins[1]<<"x"<<numbins[2]
	<<"]"<<endl;
#endif

	FLT* d_kx = d_plan->kx;
	FLT* d_ky = d_plan->ky;
	FLT* d_kz = d_plan->kz;
	CUCPX* d_c = d_plan->c;
	CUCPX* d_fw = d_plan->fine_grid_data_;

	int *d_binsize = d_plan->binsize;
	int *d_binstartpts = d_plan->binstartpts;
	int *d_numsubprob = d_plan->numsubprob;
	int *d_subprobstartpts = d_plan->subprobstartpts;
	int *d_idxnupts = d_plan->idxnupts;
	int *d_subprob_to_bin = d_plan->subprob_to_bin;
	int totalnumsubprob=d_plan->totalnumsubprob;

	FLT sigma=d_plan->spopts.upsampling_factor;
	FLT es_c=d_plan->spopts.ES_c;
	FLT es_beta=d_plan->spopts.ES_beta;
	int pirange=d_plan->spopts.pirange;
	cudaEventRecord(start);
	size_t sharedplanorysize = (bin_size_x+2*ceil(ns/2.0))*
		(bin_size_y+2*ceil(ns/2.0))*(bin_size_z+2*ceil(ns/2.0))*sizeof(CUCPX);
	if (sharedplanorysize > 49152) {
		cout<<"error: not enough shared memory"<<endl;
		return 1;
	}

	for (int t=0; t<blksize; t++) {
		if (d_plan->options_.kernel_evaluation_method == KernelEvaluationMethod::HORNER) {
			Interp_3d_Subprob_Horner<<<totalnumsubprob, 256,
				sharedplanorysize>>>(d_kx, d_ky, d_kz, d_c+t*M, d_fw+t*nf1*nf2*nf3, 
				M, ns, nf1, nf2, nf3, sigma, d_binstartpts, d_binsize, bin_size_x,
				bin_size_y, bin_size_z, d_subprob_to_bin, d_subprobstartpts,
				d_numsubprob, maxsubprobsize,numbins[0], numbins[1], numbins[2],
				d_idxnupts,pirange);
		}else{
			Interp_3d_Subprob<<<totalnumsubprob, 256,
				sharedplanorysize>>>(d_kx, d_ky, d_kz, d_c+t*M, d_fw+t*nf1*nf2*nf3, 
				M, ns, nf1, nf2, nf3, es_c, es_beta, d_binstartpts, d_binsize, 
				bin_size_x, bin_size_y, bin_size_z, d_subprob_to_bin, 
				d_subprobstartpts, d_numsubprob, maxsubprobsize,numbins[0], 
				numbins[1], numbins[2],d_idxnupts,pirange);
		}
	}
#ifdef SPREADTIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tKernel Interp_3d_Subprob (%d) \t%.3g ms\n", milliseconds,
	d_plan->options_.kernel_evaluation_method);
#endif
	return 0;
}
