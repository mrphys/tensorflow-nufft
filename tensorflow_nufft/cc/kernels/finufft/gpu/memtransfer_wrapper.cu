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
#include "memtransfer.h"

using namespace std;
using namespace tensorflow;
using namespace tensorflow::nufft;


int ALLOCGPUMEM2D_NUPTS(Plan<GPUDevice, FLT>* d_plan)
/*
	wrapper for gpu memory allocation in "setNUpts" stage.

	Melody Shih 07/25/19
*/
{
        // Mult-GPU support: set the CUDA Device ID:
        int orig_gpu_device_id;
        cudaGetDevice(& orig_gpu_device_id);
        cudaSetDevice(d_plan->options_.gpu_device_id);

	int M = d_plan->M;

	if (d_plan->sortidx ) checkCudaErrors(cudaFree(d_plan->sortidx));
	if (d_plan->idxnupts) checkCudaErrors(cudaFree(d_plan->idxnupts));

	switch(d_plan->options_.gpu_spread_method)
	{
		case GpuSpreadMethod::NUPTS_DRIVEN:
			{
				if (d_plan->options_.gpu_sort_points)
					checkCudaErrors(cudaMalloc(&d_plan->sortidx, M*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->idxnupts,M*sizeof(int)));
			}
			break;
		case GpuSpreadMethod::SUBPROBLEM:
		case GpuSpreadMethod::PAUL:
			{
				checkCudaErrors(cudaMalloc(&d_plan->idxnupts,M*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->sortidx, M*sizeof(int)));
			}
			break;
		default:
			cerr<<"err: invalid method" << endl;
	}

        // Multi-GPU support: reset the device ID
        cudaSetDevice(orig_gpu_device_id);

	return 0;
}

void FREEGPUMEMORY2D(Plan<GPUDevice, FLT>* d_plan)
/*
	wrapper for freeing gpu memory.

	Melody Shih 07/25/19
*/
{
        // Mult-GPU support: set the CUDA Device ID:
        int orig_gpu_device_id;
        cudaGetDevice(& orig_gpu_device_id);
        cudaSetDevice(d_plan->options_.gpu_device_id);

	if (!d_plan->options_.spread_interp_only) {
		checkCudaErrors(cudaFree(d_plan->fw));
		checkCudaErrors(cudaFree(d_plan->fwkerhalf1));
		checkCudaErrors(cudaFree(d_plan->fwkerhalf2));
	}
	switch(d_plan->options_.gpu_spread_method)
	{
		case GpuSpreadMethod::NUPTS_DRIVEN:
			{
				if (d_plan->options_.gpu_sort_points) {
					checkCudaErrors(cudaFree(d_plan->idxnupts));
					checkCudaErrors(cudaFree(d_plan->sortidx));
					checkCudaErrors(cudaFree(d_plan->binsize));
					checkCudaErrors(cudaFree(d_plan->binstartpts));
				}else{
					checkCudaErrors(cudaFree(d_plan->idxnupts));
				}
			}
			break;
		case GpuSpreadMethod::SUBPROBLEM:
			{
				checkCudaErrors(cudaFree(d_plan->idxnupts));
				checkCudaErrors(cudaFree(d_plan->sortidx));
				checkCudaErrors(cudaFree(d_plan->numsubprob));
				checkCudaErrors(cudaFree(d_plan->binsize));
				checkCudaErrors(cudaFree(d_plan->binstartpts));
				checkCudaErrors(cudaFree(d_plan->subprobstartpts));
				checkCudaErrors(cudaFree(d_plan->subprob_to_bin));
			}
			break;
		case GpuSpreadMethod::PAUL:
			{
				checkCudaErrors(cudaFree(d_plan->idxnupts));
				checkCudaErrors(cudaFree(d_plan->sortidx));
				checkCudaErrors(cudaFree(d_plan->numsubprob));
				checkCudaErrors(cudaFree(d_plan->binsize));
				checkCudaErrors(cudaFree(d_plan->finegridsize));
				checkCudaErrors(cudaFree(d_plan->binstartpts));
				checkCudaErrors(cudaFree(d_plan->subprobstartpts));
				checkCudaErrors(cudaFree(d_plan->subprob_to_bin));
			}
			break;
	}

	for (int i = 0; i < d_plan->options_.gpu_num_streams; i++)
		checkCudaErrors(cudaStreamDestroy(d_plan->streams[i]));

        // Multi-GPU support: reset the device ID
        cudaSetDevice(orig_gpu_device_id);
}

int ALLOCGPUMEM1D_NUPTS(Plan<GPUDevice, FLT>* d_plan)
{
	cerr<<"Not yet implemented"<<endl;
	return 1;
}
void FREEGPUMEMORY1D(Plan<GPUDevice, FLT>* d_plan)
{
	cerr<<"Not yet implemented"<<endl;
}


int ALLOCGPUMEM3D_NUPTS(Plan<GPUDevice, FLT>* d_plan)
/*
	wrapper for gpu memory allocation in "setNUpts" stage.

	Melody Shih 07/25/19
*/
{
        // Mult-GPU support: set the CUDA Device ID:
        int orig_gpu_device_id;
        cudaGetDevice(& orig_gpu_device_id);
        cudaSetDevice(d_plan->options_.gpu_device_id);

	int M = d_plan->M;

	d_plan->byte_now=0;

	if (d_plan->sortidx ) checkCudaErrors(cudaFree(d_plan->sortidx));
	if (d_plan->idxnupts) checkCudaErrors(cudaFree(d_plan->idxnupts));

	switch (d_plan->options_.gpu_spread_method)
	{
		case GpuSpreadMethod::NUPTS_DRIVEN:
			{
				if (d_plan->options_.gpu_sort_points)
					checkCudaErrors(cudaMalloc(&d_plan->sortidx, M*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->idxnupts,M*sizeof(int)));
			}
			break;
		case GpuSpreadMethod::SUBPROBLEM:
			{
				checkCudaErrors(cudaMalloc(&d_plan->idxnupts,M*sizeof(int)));
				checkCudaErrors(cudaMalloc(&d_plan->sortidx, M*sizeof(int)));
			}
			break;
		case GpuSpreadMethod::BLOCK_GATHER:
			{
				checkCudaErrors(cudaMalloc(&d_plan->sortidx,M*sizeof(int)));
			}
			break;
		default:
			cerr << "err: invalid method" << endl;
	}

        // Multi-GPU support: reset the device ID
        cudaSetDevice(orig_gpu_device_id);

	return 0;
}
void FREEGPUMEMORY3D(Plan<GPUDevice, FLT>* d_plan)
/*
	wrapper for freeing gpu memory.

	Melody Shih 07/25/19
*/
{
        // Mult-GPU support: set the CUDA Device ID:
        int orig_gpu_device_id;
        cudaGetDevice(& orig_gpu_device_id);
        cudaSetDevice(d_plan->options_.gpu_device_id);


	if (!d_plan->options_.spread_interp_only) {
		cudaFree(d_plan->fw);
		cudaFree(d_plan->fwkerhalf1);
		cudaFree(d_plan->fwkerhalf2);
		cudaFree(d_plan->fwkerhalf3);
	}

	switch (d_plan->options_.gpu_spread_method)
	{
		case GpuSpreadMethod::NUPTS_DRIVEN:
			{
				if (d_plan->options_.gpu_sort_points) {
					checkCudaErrors(cudaFree(d_plan->idxnupts));
					checkCudaErrors(cudaFree(d_plan->sortidx));
					checkCudaErrors(cudaFree(d_plan->binsize));
					checkCudaErrors(cudaFree(d_plan->binstartpts));
				}else{
					checkCudaErrors(cudaFree(d_plan->idxnupts));
				}
			}
			break;
		case GpuSpreadMethod::SUBPROBLEM:
			{
				checkCudaErrors(cudaFree(d_plan->idxnupts));
				checkCudaErrors(cudaFree(d_plan->sortidx));
				checkCudaErrors(cudaFree(d_plan->numsubprob));
				checkCudaErrors(cudaFree(d_plan->binsize));
				checkCudaErrors(cudaFree(d_plan->binstartpts));
				checkCudaErrors(cudaFree(d_plan->subprobstartpts));
				checkCudaErrors(cudaFree(d_plan->subprob_to_bin));
			}
			break;
		case GpuSpreadMethod::BLOCK_GATHER:
			{
				checkCudaErrors(cudaFree(d_plan->idxnupts));
				checkCudaErrors(cudaFree(d_plan->sortidx));
				checkCudaErrors(cudaFree(d_plan->numsubprob));
				checkCudaErrors(cudaFree(d_plan->binsize));
				checkCudaErrors(cudaFree(d_plan->binstartpts));
				checkCudaErrors(cudaFree(d_plan->subprobstartpts));
				checkCudaErrors(cudaFree(d_plan->subprob_to_bin));
			}
			break;
	}

	for (int i = 0; i < d_plan->options_.gpu_num_streams; i++)
		checkCudaErrors(cudaStreamDestroy(d_plan->streams[i]));

        // Multi-GPU support: reset the device ID
        cudaSetDevice(orig_gpu_device_id);
}
