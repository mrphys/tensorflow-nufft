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
					ier = CUINTERP2D_NUPTSDRIVEN(d_plan, blksize);
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
#ifdef SPREADTIME
	float milliseconds;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout<<"[time  ]"<< " Interp " << milliseconds <<" ms"<<endl;
#endif
	return ier;
}

