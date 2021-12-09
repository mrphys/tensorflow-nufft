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

{
        // Mult-GPU support: set the CUDA Device ID:
        int orig_gpu_device_id;
        cudaGetDevice(& orig_gpu_device_id);
        cudaSetDevice(d_plan->options_.gpu_device_id);


	d_plan->M = M;
	d_plan->num_points_ = M;


	int ier;
	ier = ALLOCGPUMEM3D_NUPTS(d_plan);

	d_plan->kx = d_kx;
	if (d_plan->rank_ > 1)
		d_plan->ky = d_ky;
	if (d_plan->rank_ > 2)
		d_plan->kz = d_kz;
	
	d_plan->points_[0] = d_kx;
	if (d_plan->rank_ > 1)
		d_plan->points_[1] = d_ky;
	if (d_plan->rank_ > 2)
		d_plan->points_[2] = d_kz;

	
		INITSPREAD(d_plan);

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
