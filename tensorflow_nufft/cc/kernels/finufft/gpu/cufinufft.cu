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
#include <complex>
#include <cufft.h>

#include "tensorflow_nufft/third_party/cuda_samples/helper_cuda.h"
#include "tensorflow_nufft/cc/kernels/finufft/gpu/cufinufft_eitherprec.h"
#include "cuspreadinterp.h"
#include "cudeconvolve.h"
#include "memtransfer.h"

using namespace std;
using namespace tensorflow;
using namespace tensorflow::nufft;



#ifdef __cplusplus
extern "C" {
#endif

int CUFINUFFT_SETPTS(int M, FLT* d_kx, FLT* d_ky, FLT* d_kz, int N, FLT *d_s,
	FLT *d_t, FLT *d_u, Plan<GPUDevice, FLT>* d_plan)
/*
	"setNUpts" stage (in single or double precision).

	In this stage, we
		(1) set the number and locations of nonuniform points
		(2) allocate gpu arrays with size determined by number of nupts
		(3) rescale x,y,z coordinates for spread/interp (on gpu, rescaled
		    coordinates are stored)
		(4) determine the spread/interp properties that only relates to the
		    locations of nupts (see 2d/spread2d_wrapper.cu,
		    3d/spread3d_wrapper.cu for what have been done in
		    function spread<rank>d_<method>_prop() )

        See ../docs/cppdoc.md for main user-facing documentation.
        Here is the old developer docs, which are useful only to translate
        the argument names from the user-facing ones:
        
	Input:
	M                 number of nonuniform points
	d_kx, d_ky, d_kz  gpu array of x,y,z locations of sources (each a size M
	                  FLT array) in [-pi, pi). set h_kz to "NULL" if dimension
	                  is less than 3. same for h_ky for dimension 1.
	N, d_s, d_t, d_u  not used for type1, type2. set to 0 and NULL.

	Input/Output:
	d_plan            pointer to a Plan<GPUDevice, FLT>. Variables and arrays inside
	                  the plan are set and allocated.

        Returned value:
        a status flag: 0 if success, otherwise an error occurred

Notes: the type FLT means either single or double, matching the
	precision of the library version called.

	Melody Shih 07/25/19; Barnett 2/16/21 moved out docs.
*/
{
        // Mult-GPU support: set the CUDA Device ID:
        int orig_gpu_device_id;
        cudaGetDevice(& orig_gpu_device_id);
        cudaSetDevice(d_plan->options_.gpu_device_id);


	int nf1 = d_plan->nf1;
	int nf2 = d_plan->nf2;
	int nf3 = d_plan->nf3;
	int rank = d_plan->rank_;

	d_plan->M = M;
#ifdef INFO
	printf("[info  ] 2d1: (ms,mt)=(%d,%d) (nf1, nf2, nf3)=(%d,%d,%d) nj=%d, ntransform = %d\n",
		d_plan->ms, d_plan->mt, d_plan->nf1, d_plan->nf2, nf3, d_plan->M,
		d_plan->num_transforms_);
#endif
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int ier;
	cudaEventRecord(start);
	switch(d_plan->rank_)
	{
		case 1:
		{
			ier = ALLOCGPUMEM1D_NUPTS(d_plan);
		}
		break;
		case 2:
		{
			ier = ALLOCGPUMEM2D_NUPTS(d_plan);
		}
		break;
		case 3:
		{
			ier = ALLOCGPUMEM3D_NUPTS(d_plan);
		}
		break;
	}
#ifdef TIME
	float milliseconds = 0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tAllocate GPU memory NUpts%.3g s\n", milliseconds/1000);
#endif

	d_plan->kx = d_kx;
	if (rank > 1)
		d_plan->ky = d_ky;
	if (rank > 2)
		d_plan->kz = d_kz;

	cudaEventRecord(start);
	switch(d_plan->rank_)
	{
		case 1:
		{
			cerr<<"Not implemented yet"<<endl;
		}
		break;
		case 2:
		{
			if (d_plan->options_.gpu_spread_method == GpuSpreadMethod::NUPTS_DRIVEN) {
				ier = CUSPREAD2D_NUPTSDRIVEN_PROP(nf1,nf2,M,d_plan);
				if (ier != 0 ) {
					printf("error: cuspread2d_nupts_prop, method(%d)\n",
						  d_plan->options_.gpu_spread_method);

                                        // Multi-GPU support: reset the device ID
                                        cudaSetDevice(orig_gpu_device_id);

					return 1;
				}
			}
			if (d_plan->options_.gpu_spread_method == GpuSpreadMethod::SUBPROBLEM) {
				ier = CUSPREAD2D_SUBPROB_PROP(nf1,nf2,M,d_plan);
				if (ier != 0 ) {
					printf("error: cuspread2d_subprob_prop, method(%d)\n",
					       d_plan->options_.gpu_spread_method);

                                        // Multi-GPU support: reset the device ID
                                        cudaSetDevice(orig_gpu_device_id);

					return 1;
				}
			}
			if (d_plan->options_.gpu_spread_method == GpuSpreadMethod::PAUL) {
				int ier = CUSPREAD2D_PAUL_PROP(nf1,nf2,M,d_plan);
				if (ier != 0 ) {
					printf("error: cuspread2d_paul_prop, method(%d)\n",
						d_plan->options_.gpu_spread_method);

                                        // Multi-GPU support: reset the device ID
                                        cudaSetDevice(orig_gpu_device_id);

					return 1;
				}
			}
		}
		break;
		case 3:
		{
			if (d_plan->options_.gpu_spread_method == GpuSpreadMethod::BLOCK_GATHER) {
				int ier = CUSPREAD3D_BLOCKGATHER_PROP(nf1,nf2,nf3,M,d_plan);
				if (ier != 0 ) {
					printf("error: cuspread3d_blockgather_prop, method(%d)\n",
						d_plan->options_.gpu_spread_method);

                                        // Multi-GPU support: reset the device ID
                                        cudaSetDevice(orig_gpu_device_id);

					return ier;
				}
			}
			if (d_plan->options_.gpu_spread_method == GpuSpreadMethod::NUPTS_DRIVEN) {
				ier = CUSPREAD3D_NUPTSDRIVEN_PROP(nf1,nf2,nf3,M,d_plan);
				if (ier != 0 ) {
					printf("error: cuspread3d_nuptsdriven_prop, method(%d)\n",
						d_plan->options_.gpu_spread_method);

                                        // Multi-GPU support: reset the device ID
                                        cudaSetDevice(orig_gpu_device_id);

					return ier;
				}
			}
			if (d_plan->options_.gpu_spread_method == GpuSpreadMethod::SUBPROBLEM) {
				int ier = CUSPREAD3D_SUBPROB_PROP(nf1,nf2,nf3,M,d_plan);
				if (ier != 0 ) {
					printf("error: cuspread3d_subprob_prop, method(%d)\n",
						d_plan->options_.gpu_spread_method);

                                        // Multi-GPU support: reset the device ID
                                        cudaSetDevice(orig_gpu_device_id);

					return ier;
				}
			}
		}
		break;
	}
#ifdef TIME
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("[time  ] \tSetup Subprob properties %.3g s\n",
		milliseconds/1000);
#endif

        // Multi-GPU support: reset the device ID
        cudaSetDevice(orig_gpu_device_id);

	return 0;
}

int CUFINUFFT_EXECUTE(CUCPX* d_c, CUCPX* d_fk, Plan<GPUDevice, FLT>* d_plan)
/*
	"exec" stage (single and double precision versions).

	The actual transformation is done here. Type and dimension of the
	transformation are defined in d_plan in previous stages.

        See ../docs/cppdoc.md for main user-facing documentation.

	Input/Output:
	d_c   a size d_plan->M CPX array on gpu (input for Type 1; output for Type
	      2)
	d_fk  a size d_plan->ms*d_plan->mt*d_plan->mu CPX array on gpu ((input for
	      Type 2; output for Type 1)

	Notes:
        i) Here CPX is a defined type meaning either complex<float> or complex<double>
	    to match the precision of the library called.
        ii) All operations are done on the GPU device (hence the d_* names)

	Melody Shih 07/25/19; Barnett 2/16/21.
*/
{
        // Mult-GPU support: set the CUDA Device ID:
        int orig_gpu_device_id;
        cudaGetDevice(& orig_gpu_device_id);
        cudaSetDevice(d_plan->options_.gpu_device_id);

	int ier;
	switch(d_plan->rank_)
	{
		case 1:
		{
			cerr<<"Not Implemented yet"<<endl;
			ier = ERR_NOTIMPLEMENTED;
		}
		break;
		case 2:
		{
			if (d_plan->type_ == TransformType::TYPE_1)
				ier = CUFINUFFT2D1_EXEC(d_c,  d_fk, d_plan);
			if (d_plan->type_ == TransformType::TYPE_2)
				ier = CUFINUFFT2D2_EXEC(d_c,  d_fk, d_plan);
			if (d_plan->type_ == TransformType::TYPE_3) {
				cerr<<"Not Implemented yet"<<endl;
				ier = ERR_NOTIMPLEMENTED;
			}
		}
		break;
		case 3:
		{
			if (d_plan->type_ == TransformType::TYPE_1)
				ier = CUFINUFFT3D1_EXEC(d_c,  d_fk, d_plan);
			if (d_plan->type_ == TransformType::TYPE_2)
				ier = CUFINUFFT3D2_EXEC(d_c,  d_fk, d_plan);
			if (d_plan->type_ == TransformType::TYPE_3) {
				cerr<<"Not Implemented yet"<<endl;
				ier = ERR_NOTIMPLEMENTED;
			}
		}
		break;
	}

        // Multi-GPU support: reset the device ID
        cudaSetDevice(orig_gpu_device_id);

	return ier;
}

int CUFINUFFT_INTERP(CUCPX* d_c, CUCPX* d_fk, Plan<GPUDevice, FLT>* d_plan)
{
	// Mult-GPU support: set the CUDA Device ID:
	int orig_gpu_device_id;
	cudaGetDevice(& orig_gpu_device_id);
	cudaSetDevice(d_plan->options_.gpu_device_id);

	int ier;

	switch(d_plan->rank_)
	{
		case 1:
			cerr<<"Not Implemented yet"<<endl;
			ier = ERR_NOTIMPLEMENTED;
			break;
		case 2:
			ier = CUFINUFFT2D_INTERP(d_c,  d_fk, d_plan);
			break;
		case 3:
			ier = CUFINUFFT3D_INTERP(d_c,  d_fk, d_plan);
			break;
	}

	// Multi-GPU support: reset the device ID
	cudaSetDevice(orig_gpu_device_id);

	return ier;
}

int CUFINUFFT_SPREAD(CUCPX* d_c, CUCPX* d_fk, Plan<GPUDevice, FLT>* d_plan)
{
	// Mult-GPU support: set the CUDA Device ID:
	int orig_gpu_device_id;
	cudaGetDevice(& orig_gpu_device_id);
	cudaSetDevice(d_plan->options_.gpu_device_id);

	int ier;

	switch(d_plan->rank_)
	{
		case 1:
			cerr<<"Not Implemented yet"<<endl;
			ier = ERR_NOTIMPLEMENTED;
			break;
		case 2:
			ier = CUFINUFFT2D_SPREAD(d_c,  d_fk, d_plan);
			break;
		case 3:
			ier = CUFINUFFT3D_SPREAD(d_c,  d_fk, d_plan);
			break;
	}

	// Multi-GPU support: reset the device ID
	cudaSetDevice(orig_gpu_device_id);

	return ier;
}

#ifdef __cplusplus
}
#endif
