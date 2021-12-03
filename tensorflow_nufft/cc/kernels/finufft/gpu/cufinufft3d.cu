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
#include <tensorflow_nufft/cc/kernels/finufft/gpu/contrib/cuda_samples/helper_cuda.h>
#include <complex>
#include <cufft.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include <tensorflow_nufft/cc/kernels/finufft/gpu/cufinufft_eitherprec.h>
#include "tensorflow_nufft/cc/kernels/finufft/gpu/cuspreadinterp.h"
#include "tensorflow_nufft/cc/kernels/finufft/gpu/cudeconvolve.h"
#include "tensorflow_nufft/cc/kernels/finufft/gpu/memtransfer.h"
#include "tensorflow_nufft/cc/kernels/nufft_plan.h"

using namespace std;
using namespace tensorflow;
using namespace tensorflow::nufft;


int CUFINUFFT3D1_EXEC(CUCPX* d_c, CUCPX* d_fk, Plan<GPUDevice, FLT>* d_plan)
/*  
	3D Type-1 NUFFT

	This function is called in "exec" stage (See ../cufinufft.cu).
	It includes (copied from doc in finufft library)
		Step 1: spread data to oversampled regular mesh using kernel
		Step 2: compute FFT on uniform mesh
		Step 3: deconvolve by division of each Fourier mode independently by the
		        Fourier series coefficient of the kernel.

	Melody Shih 07/25/19		
*/
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	int blksize; 
	int ier;
	CUCPX* d_fkstart;
	CUCPX* d_cstart;
	for (int i=0; i*d_plan->maxbatchsize < d_plan->num_transforms_; i++) {
		blksize = min(d_plan->num_transforms_ - i*d_plan->maxbatchsize, 
			d_plan->maxbatchsize);
		d_cstart = d_c + i*d_plan->maxbatchsize*d_plan->M;
		d_fkstart = d_fk + i*d_plan->maxbatchsize*d_plan->ms*d_plan->mt*
			d_plan->mu;

		d_plan->c = d_cstart;
		d_plan->fk = d_fkstart;

		checkCudaErrors(cudaMemset(d_plan->fw,0,d_plan->maxbatchsize*
					d_plan->nf1*d_plan->nf2*d_plan->nf3*sizeof(CUCPX)));
#ifdef TIME
		float milliseconds = 0;
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tInitialize fw\t\t %.3g s\n", milliseconds/1000);
#endif
		// Step 1: Spread
		cudaEventRecord(start);
		ier = CUSPREAD3D(d_plan, blksize);
		if (ier != 0 ) {
			printf("error: cuspread3d, method(%d)\n", d_plan->options_.gpu_spread_method);
			return ier;
		}
#ifdef TIME
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tSpread (%d)\t\t %.3g s\n", milliseconds/1000, 
			d_plan->options_.gpu_spread_method);
#endif
		// Step 2: FFT
		cudaEventRecord(start);
		cufftResult result = CUFFT_EX(
			d_plan->fftplan, d_plan->fw, d_plan->fw, static_cast<int>(d_plan->fft_direction_));
		if (result != CUFFT_SUCCESS) {
			fprintf(stderr,"[%s] CUFFT_EX failed with error code: %d\n",__func__,result);
    		return ERR_CUFFT;
		}
#ifdef TIME
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tCUFFT Exec\t\t %.3g s\n", milliseconds/1000);
#endif

		// Step 3: deconvolve and shuffle
		cudaEventRecord(start);
		CUDECONVOLVE3D(d_plan, blksize);
#ifdef TIME
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tDeconvolve\t\t %.3g s\n", milliseconds/1000);
#endif
	}
	return ier;
}

int CUFINUFFT3D2_EXEC(CUCPX* d_c, CUCPX* d_fk, Plan<GPUDevice, FLT>* d_plan)
/*  
	3D Type-2 NUFFT

	This function is called in "exec" stage (See ../cufinufft.cu).
	It includes (copied from doc in finufft library)
		Step 1: deconvolve (amplify) each Fourier mode, dividing by kernel 
		        Fourier coeff
		Step 2: compute FFT on uniform mesh
		Step 3: interpolate data to regular mesh

	Melody Shih 07/25/19		
*/
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int blksize;
	int ier;
	CUCPX* d_fkstart;
	CUCPX* d_cstart;
	for (int i=0; i*d_plan->maxbatchsize < d_plan->num_transforms_; i++) {
		blksize = min(d_plan->num_transforms_ - i*d_plan->maxbatchsize, 
			d_plan->maxbatchsize);
		d_cstart  = d_c  + i*d_plan->maxbatchsize*d_plan->M;
		d_fkstart = d_fk + i*d_plan->maxbatchsize*d_plan->ms*d_plan->mt*
			d_plan->mu;

		d_plan->c = d_cstart;
		d_plan->fk = d_fkstart;

		// Step 1: amplify Fourier coeffs fk and copy into upsampled array fw
		cudaEventRecord(start);
		CUDECONVOLVE3D(d_plan, blksize);
#ifdef TIME
		float milliseconds = 0;
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tAmplify & Copy fktofw\t %.3g s\n", milliseconds/1000);
#endif
		// Step 2: FFT
		cudaEventRecord(start);
		cudaDeviceSynchronize();
		cufftResult result = CUFFT_EX(
			d_plan->fftplan, d_plan->fw, d_plan->fw, static_cast<int>(d_plan->fft_direction_));
		if (result != CUFFT_SUCCESS) {
			fprintf(stderr,"[%s] CUFFT_EX failed with error code: %d\n",__func__,result);
    		return ERR_CUFFT;
		}
#ifdef TIME
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tCUFFT Exec\t\t %.3g s\n", milliseconds/1000);
#endif

		// Step 3: deconvolve and shuffle
		cudaEventRecord(start);
		ier = CUINTERP3D(d_plan, blksize);
		if (ier != 0 ) {
			printf("error: cuinterp3d, method(%d)\n", d_plan->options_.gpu_spread_method);
			return ier;
		}
#ifdef TIME
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("[time  ] \tUnspread (%d)\t\t %.3g s\n", milliseconds/1000,
			d_plan->options_.gpu_spread_method);
#endif
	}

	return ier;
}

int CUFINUFFT3D_INTERP(CUCPX* d_c, CUCPX* d_fk, Plan<GPUDevice, FLT>* d_plan)
{
	assert(d_plan->spopts.spread_direction == SpreadDirection::INTERP);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	int blksize;
	int ier;
	int gridsize = d_plan->ms*d_plan->mt*d_plan->mu;
	CUCPX* d_fkstart;
	CUCPX* d_cstart;
	
	for (int i=0; i*d_plan->maxbatchsize < d_plan->num_transforms_; i++) {
		blksize = min(d_plan->num_transforms_ - i*d_plan->maxbatchsize, 
			d_plan->maxbatchsize);
		d_cstart  = d_c  + i*d_plan->maxbatchsize*d_plan->M;
		d_fkstart = d_fk + i*d_plan->maxbatchsize*gridsize;

		d_plan->c = d_cstart;
		d_plan->fw = d_fkstart;

		cudaEventRecord(start);
		ier = CUINTERP3D(d_plan, blksize);
		if (ier != 0 ) {
			printf("error: cuinterp3d, method(%d)\n", d_plan->options_.gpu_spread_method);
			return ier;
		}
	}

	using namespace thrust::placeholders;
	thrust::device_ptr<FLT> dev_ptr((FLT*) d_c);
	thrust::transform(dev_ptr, dev_ptr + 2*d_plan->num_transforms_*d_plan->M,
					  dev_ptr, _1 * (FLT) d_plan->spopts.ES_scale); 

	return ier;
}

int CUFINUFFT3D_SPREAD(CUCPX* d_c, CUCPX* d_fk, Plan<GPUDevice, FLT>* d_plan)
{
	assert(d_plan->spopts.spread_direction == SpreadDirection::SPREAD);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	int blksize;
	int ier;
	int gridsize = d_plan->ms*d_plan->mt*d_plan->mu;
	CUCPX* d_fkstart;
	CUCPX* d_cstart;
	for (int i=0; i*d_plan->maxbatchsize < d_plan->num_transforms_; i++) {
		blksize = min(d_plan->num_transforms_ - i*d_plan->maxbatchsize, 
			d_plan->maxbatchsize);
		d_cstart   = d_c + i*d_plan->maxbatchsize*d_plan->M;
		d_fkstart = d_fk + i*d_plan->maxbatchsize*gridsize;

		d_plan->c  = d_cstart;
		d_plan->fw = d_fkstart;

		cudaEventRecord(start);
		ier = CUSPREAD3D(d_plan,blksize);
		if (ier != 0 ) {
			printf("error: cuspread3d, method(%d)\n", d_plan->options_.gpu_spread_method);
			return ier;
		}
	}

	using namespace thrust::placeholders;
	thrust::device_ptr<FLT> dev_ptr((FLT*) d_fk);
	thrust::transform(dev_ptr, dev_ptr + 2*d_plan->num_transforms_*gridsize,
					  dev_ptr, _1 * (FLT) d_plan->spopts.ES_scale); 
	
	return ier;
}
